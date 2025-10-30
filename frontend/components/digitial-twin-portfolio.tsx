'use client'
import React, { useState, useEffect, useRef } from 'react';
import { Briefcase, Award, FileText, Linkedin, Github, MessageSquare, X, ExternalLink, RefreshCw, Send, User, Bot } from 'lucide-react';

interface Message {
  id: string;
  sender: 'user' | 'assistant' | 'bot';
  text: string;
  content: string;
  timestamp: Date;
}


export default function DigitalTwin() {
  const [isTalking, setIsTalking] = useState(false);
  const [showResumeModal, setShowResumeModal] = useState(false);
  const [showChat, setShowChat] = useState(false);
  const [activeSection, setActiveSection] = useState('about');
  const [formData, setFormData] = useState({ name: '', email: '', message: '' });
  const [isRestarting, setIsRestarting] = useState(false);
  const [input, setInput] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [messages, setMessages] = useState<Message[]>([
    {
      id: Date.now().toString(),
      text: "Hello! I'm Darin's Digital Twin. Ask me about my experience with AI/ML development and deployment, cloud architecture, or any of my projects!",
      sender: 'bot',
      content: input,
      timestamp: new Date()
    }
  ]);
  
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = React.useRef<HTMLDivElement>(null);
  const inputRef = React.useRef<HTMLInputElement>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string>('');

  useEffect(() => {
    const interval = setInterval(() => {
      setIsTalking(prev => !prev);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (showChat && messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, showChat]);


  const handleSendMessage = async () => {
    if (!inputText.trim() || isLoading) return;

    const newMessage: Message = {
      id: Date.now().toString(),
      content: inputText,
      text: inputText,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, newMessage]);
    // simulateResponse(inputText);
    setInputText('');
    setIsLoading(true)

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/chat`, {
          method: 'POST',
          headers: {
              'Content-Type': 'application/json',
          },
          body: JSON.stringify({
              message: newMessage.content,
              session_id: sessionId || undefined,
          }),
      });

      if (!response.ok) throw new Error('Failed to send message');

      const data = await response.json();

      if (!sessionId) {
          setSessionId(data.session_id);
      }

      const assistantMessage: Message = {
          id: (Date.now() + 1).toString(),
          sender: 'assistant',
          content: data.response,
          text: data.response,
          timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage: Message = {
          id: (Date.now() + 1).toString(),
          sender: 'assistant',
          content: 'Sorry, I encountered an error. Please try again.',
          text: 'Sorry, I encountered an error. Please try again.',
          timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
        setIsLoading(false);
        // Refocus the input after message is sent
        setTimeout(() => {
            inputRef.current?.focus();
        }, 100);
    }

  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleRestartSpace = () => {
    setIsRestarting(true);
    setTimeout(() => {
      setIsRestarting(false);
      alert('Hugging Face space restarted successfully!');
    }, 2000);
  };

    // Handle form input changes
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
        ...prev,
        [name]: value
    }));
    };

  const handleSubmitResume = async () => {
    if (!formData.name || !formData.email) {
      alert('Please fill in your name and email');
      return;
    }

    // Basic email validation
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(formData.email)) {
    alert('Please enter a valid email address');
    return;
  }

  
    try {
      // Show loading state (you can add a spinner if you have one)
      setIsSubmitting(true); // Add this state: const [isSubmitting, setIsSubmitting] = useState(false);
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/send-resume-request`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: formData.name,
          email: formData.email,
          message: formData.message || ''
        })
      });
  
      const data = await response.json();
  
      if (response.ok && data.success) {
        // Success
        alert(`✅ Resume request sent successfully!\n\nWe'll send the resume to ${formData.email} shortly.`);
        setShowResumeModal(false);
        setFormData({ name: '', email: '', message: '' });
      } else {
        // Error from backend
        alert(`❌ Failed to send request: ${data.detail || data.message || 'Unknown error'}`);
      }
    } catch (error) {
      // Network or other error
      console.error('Error sending resume request:', error);
      alert('❌ Failed to send resume request. Please try again or contact directly.');
    } finally {
      setIsSubmitting(false);
    }
  };

  // Handle Enter key in resume form
  const handleResumeFormKeyPress = (e: React.KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    // Only trigger submit on Enter for non-textareas
    if (
      e.key === 'Enter' &&
      (e.target as HTMLElement).tagName !== 'TEXTAREA'
    ) {
      e.preventDefault();
      handleSubmitResume();
    }
  };

  // Check if avatar exists
  const [hasAvatar, setHasAvatar] = useState(false);
  useEffect(() => {
      // Check if avatar.png exists
      fetch('/avatar.png', { method: 'HEAD' })
          .then(res => setHasAvatar(res.ok))
          .catch(() => setHasAvatar(false));
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-20 w-96 h-96 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse"></div>
        <div className="absolute bottom-20 right-20 w-96 h-96 bg-blue-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse" style={{animationDelay: '1s'}}></div>
      </div>

      <header className="relative z-10 border-b border-white/10 backdrop-blur-sm bg-white/5">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              AI in Production
            </h1>
            <p className="text-sm text-gray-400">Digital Twin to the Cloud</p>
          </div>
          <div className="flex gap-4">
            <a href="https://www.linkedin.com/in/darinjeswilliams/" target="_blank" rel="noopener noreferrer" className="p-2 rounded-lg hover:bg-white/10 transition-all">
              <Linkedin className="w-5 h-5" />
            </a>
            <a href="https://github.com/darinjswilliams" target="_blank" rel="noopener noreferrer" className="p-2 rounded-lg hover:bg-white/10 transition-all">
              <Github className="w-5 h-5" />
            </a>
          </div>
        </div>
      </header>

      <main className="relative z-10 max-w-7xl mx-auto px-6 py-12">
        <div className="mb-16">
          <div className="flex flex-col items-center text-center mb-12">
            <div className="relative mb-8">
              <div className={`absolute inset-0 rounded-full bg-gradient-to-r from-blue-500 to-purple-500 blur-xl transition-all duration-500 ${isTalking ? 'opacity-70 scale-110' : 'opacity-30 scale-100'}`}></div>
              <div className={`relative w-48 h-48 rounded-full border-4 transition-all duration-500 ${isTalking ? 'border-purple-400 scale-105' : 'border-blue-400'} overflow-hidden bg-slate-700`}>
                <div className="w-full h-full flex items-center justify-center">
                  <div className="text-center">
                    <div className="text-4xl mb-2">👤</div>
                    {hasAvatar ? (
                            <img 
                                src="/avatar.png" 
                                alt="Digital Twin Avatar" 
                                className="w-55 h-55 rounded-full mx-auto mb-3 border-2 border-gray-300"
                            />
                        ) : (
                            <Bot className="w-12 h-12 mx-auto mb-3 text-gray-400" />
                        )}
                  </div>
                </div>
              </div>
              {isTalking && (
                <div className="absolute -bottom-2 left-1/2 transform -translate-x-1/2 flex gap-1">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                </div>
              )}
            </div>

            <h2 className="text-4xl font-bold mb-4 bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
              Hello! I'm Darin's Digital Twin
            </h2>
            <p className="text-xl text-gray-300 mb-8 max-w-2xl">
              Ask me anything about AI deployment, cloud architecture, and production-ready solutions.
              I'm here to represent my expertise and help you understand my capabilities.
            </p>

            <div className="flex flex-wrap gap-4 justify-center">
              <button
                onClick={() => setShowResumeModal(true)}
                className="px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg font-semibold hover:scale-105 transition-transform flex items-center gap-2"
              >
                <FileText className="w-5 h-5" />
                Request Resume
              </button>
            </div>
          </div>

          <div className="flex justify-center gap-4 mb-8 flex-wrap">
            {['about', 'experience', 'certifications', 'projects'].map((section) => (
              <button
                key={section}
                onClick={() => setActiveSection(section)}
                className={`px-6 py-3 rounded-lg font-semibold transition-all ${
                  activeSection === section
                    ? 'bg-gradient-to-r from-blue-500 to-purple-500'
                    : 'bg-white/10 hover:bg-white/20'
                }`}
              >
                {section.charAt(0).toUpperCase() + section.slice(1)}
              </button>
            ))}
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            {activeSection === 'about' && (
              <>
                <div className="bg-white/10 backdrop-blur-md rounded-2xl p-8 border border-white/20 hover:border-purple-400 transition-all hover:scale-105">
                  <div className="flex items-center gap-3 mb-4">
                    <Briefcase className="w-8 h-8 text-purple-400" />
                    <h3 className="text-2xl font-bold">Professional Profile</h3>
                  </div>
                  <p className="text-gray-300 mb-4">
                    AI/ML Engineer specializing in production-ready solutions, cloud architecture, and scalable deployments.
                  </p>
                  <div className="flex flex-wrap gap-2">
                    <span className="px-3 py-1 bg-blue-500/20 rounded-full text-sm">TypeScript</span>
                    <span className="px-3 py-1 bg-purple-500/20 rounded-full text-sm">FastAPI</span>
                    <span className="px-3 py-1 bg-pink-500/20 rounded-full text-sm">AI/ML</span>
                    <span className="px-3 py-1 bg-green-500/20 rounded-full text-sm">Cloud</span>
                  </div>
                </div>

                <div className="bg-white/10 backdrop-blur-md rounded-2xl p-8 border border-white/20 hover:border-blue-400 transition-all hover:scale-105">
                  <div className="flex items-center gap-3 mb-4">
                    <MessageSquare className="w-8 h-8 text-blue-400" />
                    <h3 className="text-2xl font-bold">Interactive AI</h3>
                  </div>
                  <p className="text-gray-300 mb-4">
                    This digital twin is powered by advanced AI models, capable of answering questions about my experience, skills, and projects in real-time.
                  </p>
                  <button className="px-4 py-2 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg hover:scale-105 transition-transform" onClick={() => setShowChat(true)}>
                    Start Conversation
                  </button>
                </div>
              </>
            )}

            {activeSection === 'experience' && (
              <div className="md:col-span-2 bg-white/10 backdrop-blur-md rounded-2xl p-8 border border-white/20">
                <h3 className="text-2xl font-bold mb-6">Professional Experience</h3>
                <div className="space-y-6">
                  <div className="border-l-4 border-purple-400 pl-4">
                    <h4 className="text-xl font-semibold">Lead Software Engineer</h4>
                    <p className="text-gray-400">BlueCross BlueShield of Texas</p>
                    <p className="text-gray-300 mt-2">Architected microservices 1M+ daily transactions</p>

                  </div>
                  <div className="border-l-4 border-blue-400 pl-4">
                    <h4 className="text-xl font-semibold">Senior Sotfware Engineer</h4>
                    <p className="text-gray-400">BlueCross BlueShield of Texas </p>
                    <p className="text-gray-300 mt-2">Created Reusable ReactJS Components</p>
                  </div>
                </div>
              </div>
            )}

            {activeSection === 'certifications' && (
              <div className="md:col-span-2 bg-white/10 backdrop-blur-md rounded-2xl p-8 border border-white/20">
                <div className="flex items-center gap-3 mb-6">
                  <Award className="w-8 h-8 text-yellow-400" />
                  <h3 className="text-2xl font-bold">Certifications</h3>
                </div>
                <div className="grid md:grid-cols-3 gap-4">
                  <div className="bg-white/5 p-4 rounded-lg border border-white/10 hover:border-yellow-400 transition-all">
                    <h4 className="font-semibold mb-2">Azure AI Engineer</h4>
                    <p className="text-sm text-gray-400">Microsoft</p>
                  </div>
                  <div className="bg-white/5 p-4 rounded-lg border border-white/10 hover:border-yellow-400 transition-all">
                    <h4 className="font-semibold mb-2">AI Stratergies & Roadmap</h4>
                    <p className="text-sm text-gray-400">MIT</p>
                  </div>
                  <div className="bg-white/5 p-4 rounded-lg border border-white/10 hover:border-yellow-400 transition-all">
                    <h4 className="font-semibold mb-2">Professional AI/ML Engineer</h4>
                    <p className="text-sm text-gray-400">Microsoft</p>
                  </div>
                </div>
              </div>
            )}

            {activeSection === 'projects' && (
              <div className="md:col-span-2 bg-white/10 backdrop-blur-md rounded-2xl p-8 border border-white/20">
                <h3 className="text-2xl font-bold mb-6">Featured Projects</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-white/5 p-6 rounded-lg border border-white/10 hover:border-purple-400 transition-all group">
                    <a
                      href='https://huggingface.co/spaces/darinjswilliams/superkartfrontend'
                      target='_blank'
                      rel='noopener noreferrer'
                      className='group'
                      >
                    <h4 className="font-semibold mb-2 flex items-center gap-2">
                      Time Series Forecasting
                      <ExternalLink className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity" />
                    </h4>
                    </a>
                    <p className="text-sm text-gray-400">National Retail Chain</p>
                  </div>
                  <div className="bg-white/5 p-6 rounded-lg border border-white/10 hover:border-blue-400 transition-all group">
                  <a
                      href='https://huggingface.co/spaces/darinjswilliams/customerchurn'
                      target='_blank'
                      rel='noopener noreferrer'
                      className='group'
                      >
                    <h4 className="font-semibold mb-2 flex items-center gap-2">
                      Customer Churn
                      <ExternalLink className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity" />
                    </h4>
                    </a>
                    <p className="text-sm text-gray-400">Scalable production systems</p>
                  </div>
                  <div className="bg-white/5 p-6 rounded-lg border border-white/10 hover:border-blue-400 transition-all group">
                  <a
                      href='https://drive.google.com/file/d/16akHpzVdFvSikzmS7Y3IAb63dJXPYM03/view'
                      target='_blank'
                      rel='noopener noreferrer'
                      className='group'
                      >
                    <h4 className="font-semibold mb-2 flex items-center gap-2">
                      Soccer Analysis
                      <ExternalLink className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity" />
                    </h4>
                    </a>
                    <p className="text-sm text-gray-400">YOLO - Sports Analysis</p>
                  </div>
                  <div className="bg-white/5 p-6 rounded-lg border border-white/10 hover:border-blue-400 transition-all group">
                  <a
                      href='https://github.com/darinjswilliams/medical-rag'
                      target='_blank'
                      rel='noopener noreferrer'
                      className='group'
                      >
                    <h4 className="font-semibold mb-2 flex items-center gap-2">
                       Agentic RAG Medical Assistant
                      <ExternalLink className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity" />
                    </h4>
                    </a>
                    <p className="text-sm text-gray-400">Agentic RAG</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      {showChat && (
        <div className="fixed inset-0 z-50 flex items-center justify-end">
          <div 
            className="absolute inset-0 bg-black/50 backdrop-blur-sm"
            onClick={() => setShowChat(false)}
          ></div>
          
          <div className="relative w-full max-w-md h-full bg-slate-900 shadow-2xl flex flex-col animate-slide-in">
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-4 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-white/20 flex items-center justify-center">
                  <Bot className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="font-semibold">Digital Twin AI</h3>
                  <p className="text-xs text-gray-200">Online</p>
                </div>
              </div>
              <button
                onClick={() => setShowChat(false)}
                className="p-2 hover:bg-white/10 rounded-lg transition-all"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex gap-2 ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  {msg.sender === 'bot' && (
                    <div className="w-8 h-8 rounded-full bg-purple-500 flex items-center justify-center flex-shrink-0">
                      <Bot className="w-5 h-5" />
                    </div>
                  )}
                  <div
                    className={`max-w-xs px-4 py-2 rounded-2xl ${
                      msg.sender === 'user'
                        ? 'bg-blue-600 text-white'
                        : 'bg-white/10 text-gray-100'
                    }`}
                  >
                    <p className="text-sm">{msg.text}</p>
                    <p className="text-xs opacity-60 mt-1">
                      {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </p>
                  </div>
                  {msg.sender === 'user' && (
                    <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center flex-shrink-0">
                      <User className="w-5 h-5" />
                    </div>
                  )}
                </div>
              ))}
              
              {isTyping && (
                <div className="flex gap-2 justify-start">
                  <div className="w-8 h-8 rounded-full bg-purple-500 flex items-center justify-center flex-shrink-0">
                    <Bot className="w-5 h-5" />
                  </div>
                  <div className="bg-white/10 px-4 py-3 rounded-2xl">
                    <div className="flex gap-1">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            <div className="p-4 border-t border-white/10">
              <div className="flex gap-2">
                <input
                  ref={inputRef}
                  type="text"
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Type your message..."
                  disabled={isLoading}
                  autoFocus
                  className="flex-1 px-4 py-2 bg-white/10 border border-white/20 rounded-lg focus:outline-none focus:border-purple-400 text-white"
                />
                <button
                  onClick={handleSendMessage}
                  disabled={!inputText.trim()}
                  className="px-4 py-2 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg hover:scale-105 transition-transform disabled:opacity-50 disabled:hover:scale-100"
                >
                  <Send className="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {showResumeModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
          <div className="bg-slate-800 rounded-2xl p-8 max-w-md w-full border border-white/20 relative">
            <button
              onClick={() => setShowResumeModal(false)}
              className="absolute top-4 right-4 p-2 hover:bg-white/10 rounded-lg transition-all"
            >
              <X className="w-5 h-5" />
            </button>
            <h3 className="text-2xl font-bold mb-6">Request Resume</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">Name</label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({...formData, name: e.target.value})}
                  onKeyUp={handleResumeFormKeyPress}
                  className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg focus:outline-none focus:border-purple-400 text-white"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Email</label>
                <input
                  type="email"
                  value={formData.email}
                  onChange={(e) => setFormData({...formData, email: e.target.value})}
                  onKeyPress={handleResumeFormKeyPress}
                  className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg focus:outline-none focus:border-purple-400 text-white"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Message (Optional)</label>
                <textarea
                  value={formData.message}
                  onChange={(e) => setFormData({...formData, message: e.target.value})}
                  rows={3}
                  className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg focus:outline-none focus:border-purple-400 text-white"
                />
              </div>
              <button
                onClick={handleSubmitResume}
                className="w-full px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg font-semibold hover:scale-105 transition-transform"
              >
                Send Request
              </button>
            </div>
          </div>
        </div>
      )}

      <style>{`
        @keyframes slide-in {
          from {
            transform: translateX(100%);
          }
          to {
            transform: translateX(0);
          }
        }
        .animate-slide-in {
          animation: slide-in 0.3s ease-out;
        }
      `}</style>
    </div>
  );
}