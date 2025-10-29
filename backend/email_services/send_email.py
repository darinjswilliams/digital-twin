import os
from fastapi import HTTPException
from typing import Dict
from datetime import datetime
import requests
from dotenv import load_dotenv

load_dotenv()

BREVO_API_KEY = os.getenv("BREVO_API_KEY", "")
SENDER_EMAIL = os.getenv("SENDER_EMAIL",  "")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL", "")
BREVO_API_URL = os.getenv("BREVO_API_URL","")
SENDER_NAME = os.getenv("SENDER_NAME", "")

def send_email_brevo(name: str, email: str, user_message: str) -> Dict:
    """Send email using Brevo API"""
    
    if not BREVO_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="Email configuration missing. Set BREVO_API_KEY in .env"
        )
    
    if not SENDER_EMAIL or not RECIPIENT_EMAIL:
        raise HTTPException(
            status_code=500,
            detail="Email configuration missing. Set SENDER_EMAIL and RECIPIENT_EMAIL in .env"
        )
    
    # Ensure all inputs are clean strings
    name = str(name).strip()
    email = str(email).strip()
    user_message = str(user_message).strip() if user_message else ""
    
    # Get current timestamp as string
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    # Create HTML email content
    html_content = f"""<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background-color: #4F46E5; color: white; padding: 20px; text-align: center; }}
        .content {{ background-color: #f9f9f9; padding: 20px; margin-top: 20px; }}
        .field {{ margin-bottom: 15px; }}
        .label {{ font-weight: bold; color: #4F46E5; }}
        .footer {{ margin-top: 20px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>Resume Request from Digital Twin</h2>
        </div>
        <div class="content">
            <div class="field">
                <span class="label">Name:</span> {name}
            </div>
            <div class="field">
                <span class="label">Email:</span> {email}
            </div>
            <div class="field">
                <span class="label">Message:</span>
                <p>{user_message if user_message else 'No additional message provided'}</p>
            </div>
        </div>
        <div class="footer">
            <p>Sent at: {timestamp}</p>
        </div>
    </div>
</body>
</html>"""
    
    # Plain text version
    text_content = f"""Resume Request

Name: {name}
Email: {email}
Message: {user_message if user_message else 'No additional message provided'}

Sent at: {timestamp}"""
    
    # Build the payload as a simple dict
    payload = {
        "sender": {
            "name": str(SENDER_NAME),
            "email": str(SENDER_EMAIL)
        },
        "to": [
            {
                "email": str(RECIPIENT_EMAIL),
                "name": "Recipient"
            }
        ],
        "subject": f"Resume Request from {name}",
        "htmlContent": html_content,
        "textContent": text_content
    }
    
    # Headers
    headers = {
        "api-key": str(BREVO_API_KEY),
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    try:
        print(f"Sending email to Brevo API...")
        print(f"Sender: {SENDER_EMAIL}")
        print(f"Recipient: {RECIPIENT_EMAIL}")
        
        # Make the request
        response = requests.post(
            BREVO_API_URL,
            json=payload,
            headers=headers,
            timeout=10
        )
        
        print(f"Brevo response status: {response.status_code}")
        
        # Check if request was successful
        if response.status_code == 201:
            try:
                response_data = response.json()
                message_id = response_data.get("messageId", "unknown")
            except:
                message_id = "unknown"
            
            print(f"Email sent successfully! Message ID: {message_id}")
            
            # Return a simple dict
            return {
                "success": True,
                "message": "Email sent successfully",
                "message_id": str(message_id)
            }
        else:
            # Handle error response
            try:
                error_data = response.json()
                error_message = error_data.get("message", "Unknown error")
            except:
                error_message = response.text or "Unknown error"
            
            print(f"Brevo API error: {response.status_code} - {error_message}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to send email: {error_message}"
            )
            
    except requests.exceptions.Timeout:
        print("Brevo API request timed out")
        raise HTTPException(status_code=504, detail="Email service timeout - please try again")
    
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error to Brevo API: {e}")
        raise HTTPException(status_code=503, detail="Could not connect to email service")
    
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    
    except HTTPException:
        # Re-raise HTTPExceptions
        raise
    
    except Exception as e:
        print(f"Unexpected error in send_email_brevo: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Email error: {str(e)}")