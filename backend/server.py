from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
import os
from dotenv import load_dotenv
from typing import Optional, List, Dict
import json
import uuid
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
from context import prompt
from email_services import send_email_brevo

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure CORS
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize Bedrock client
bedrock_client = boto3.client(
    service_name="bedrock-runtime", 
    region_name=os.getenv("DEFAULT_AWS_REGION", "us-east-1")
)

# Bedrock model selection
# Available models:
# - amazon.nova-micro-v1:0  (fastest, cheapest)
# - amazon.nova-lite-v1:0   (balanced - default)
# - amazon.nova-pro-v1:0    (most capable, higher cost)
# Remember the Heads up: you might need to add us. or eu. prefix to the below model id
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")

# Memory storage configuration
USE_S3 = os.getenv("USE_S3", "false").lower() == "true"
S3_BUCKET = os.getenv("S3_BUCKET", "")
MEMORY_DIR = os.getenv("MEMORY_DIR", "../memory")

# Initialize S3 client if needed
if USE_S3:
    s3_client = boto3.client("s3")


# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


class Message(BaseModel):
    role: str
    content: str
    timestamp: str

class ResumeRequest(BaseModel):
    name: str
    email: EmailStr
    message: Optional[str] = ""

class EmailResponse(BaseModel):
    success: bool
    message: str
    message_id: Optional[str] = None


# Memory management functions
def get_memory_path(session_id: str) -> str:
    return f"{session_id}.json"


def load_conversation(session_id: str) -> List[Dict]:
    """Load conversation history from storage"""
    if USE_S3:
        try:
            response = s3_client.get_object(Bucket=S3_BUCKET, Key=get_memory_path(session_id))
            return json.loads(response["Body"].read().decode("utf-8"))
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return []
            raise
    else:
        # Local file storage
        file_path = os.path.join(MEMORY_DIR, get_memory_path(session_id))
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return json.load(f)
        return []


def save_conversation(session_id: str, messages: List[Dict]):
    """Save conversation history to storage"""
    if USE_S3:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=get_memory_path(session_id),
            Body=json.dumps(messages, indent=2),
            ContentType="application/json",
        )
    else:
        # Local file storage
        os.makedirs(MEMORY_DIR, exist_ok=True)
        file_path = os.path.join(MEMORY_DIR, get_memory_path(session_id))
        with open(file_path, "w") as f:
            json.dump(messages, f, indent=2)

def call_bedrock(conversation: List[Dict], user_message: str) -> str:
    """Call AWS Bedrock with conversation history"""
    
    # Get system prompt
    system_prompt = prompt()
    
    # Build messages in Bedrock format
    messages = []
    
    # If this is the first message, incorporate system prompt
    if not conversation:
        # First message: combine system prompt with user message
        combined_message = f"{system_prompt}\n\n{user_message}"
        messages.append({
            "role": "user",
            "content": [{"text": combined_message}]
        })
    else:
        # Add conversation history (last 20 messages = 10 exchanges)
        for msg in conversation[-20:]:
            role = msg.get("role")
            content = msg.get("content", "")
            
            # Skip if role is invalid
            if role not in ["user", "assistant"]:
                continue
            
            messages.append({
                "role": role,
                "content": [{"text": str(content)}]
            })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": [{"text": user_message}]
        })
    
    # Ensure messages alternate between user and assistant
    # Bedrock requires: user -> assistant -> user -> assistant
    fixed_messages = []
    last_role = None
    
    for msg in messages:
        current_role = msg["role"]
        
        if last_role == current_role:
            # Same role twice in a row - merge messages
            if fixed_messages:
                fixed_messages[-1]["content"][0]["text"] += "\n\n" + msg["content"][0]["text"]
        else:
            fixed_messages.append(msg)
            last_role = current_role
    
    # Must start with user
    if fixed_messages and fixed_messages[0]["role"] != "user":
        fixed_messages.insert(0, {
            "role": "user",
            "content": [{"text": system_prompt}]
        })
    
    # Must end with user (for Bedrock to respond)
    if fixed_messages and fixed_messages[-1]["role"] != "user":
        fixed_messages.append({
            "role": "user",
            "content": [{"text": user_message}]
        })
    
    try:
        print(f"Sending {len(fixed_messages)} messages to Bedrock")
        
        # Call Bedrock using the converse API
        response = bedrock_client.converse(
            modelId=BEDROCK_MODEL_ID,
            messages=fixed_messages,
            inferenceConfig={
                "maxTokens": 2000,
                "temperature": 0.7,
                "topP": 0.9
            }
        )
        
        # Extract the response text
        return response["output"]["message"]["content"][0]["text"]
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error'].get('Message', str(e))
        
        print(f"Bedrock error: {error_code} - {error_message}")
        
        # Log the messages that caused the error
        print("Messages sent to Bedrock:")
        for i, msg in enumerate(fixed_messages):
            print(f"  {i}: {msg['role']} - {msg['content'][0]['text'][:100]}...")
        
        if error_code == 'ValidationException':
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid message format for Bedrock: {error_message}"
            )
        elif error_code == 'AccessDeniedException':
            raise HTTPException(
                status_code=403, 
                detail="Access denied to Bedrock model. Check AWS credentials and model access."
            )
        elif error_code == 'ResourceNotFoundException':
            raise HTTPException(
                status_code=404,
                detail=f"Bedrock model not found: {BEDROCK_MODEL_ID}"
            )
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Bedrock error: {error_message}"
            )
            
@app.get("/")
async def root():
    return {
        "message": "AI Digital Twin API",
        "memory_enabled": True,
        "storage": "S3" if USE_S3 else "local",
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "use_s3": USE_S3, "bedrock_model": BEDROCK_MODEL_ID}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())

        # Load conversation history
        conversation = load_conversation(session_id)

        # Call Bedrock for response
        assistant_response = call_bedrock(conversation, request.message)

       # Update conversation history
        conversation.append(
            {"role": "user", "content": request.message, "timestamp": datetime.now().isoformat()}
        )
        conversation.append(
            {
                "role": "assistant",
                "content": assistant_response,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Save conversation
        save_conversation(session_id, conversation)

        return ChatResponse(response=assistant_response, session_id=session_id)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversation/{session_id}")
async def get_conversation(session_id: str):
    """Retrieve conversation history"""
    try:
        conversation = load_conversation(session_id)
        return {"session_id": session_id, "messages": conversation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/send-resume-request", response_model=EmailResponse)
async def send_resume_request(request: ResumeRequest):
    """
    Endpoint to handle resume requests and send email notifications via Brevo
    """
    try:
        # Send email via Brevo
        result = send_email_brevo(
            name=request.name,
            email=request.email,
            user_message=request.message or ""
        )
        
        # Log the request
        log_data = {
            "name": request.name,
            "email": request.email,
            "message": request.message,
            "timestamp": datetime.now().isoformat(),
            "message_id": result.get("message_id")
        }
        print(f"Resume request logged: {log_data}")
        
        return EmailResponse(
            success=True,
            message="Resume request sent successfully!",
            message_id=result.get("message_id")
        )
        
    except HTTPException as he:
        # Re-raise HTTP exceptions as-is
        print(f"HTTP Exception in send_resume_request: {he.detail}")
        raise
    except Exception as e:
        # Catch any unexpected errors
        error_msg = str(e)
        print(f"Unexpected error in send_resume_request: {error_msg}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Server error: {error_msg}"
        )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)