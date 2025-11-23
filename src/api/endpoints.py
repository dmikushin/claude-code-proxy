from fastapi import APIRouter, HTTPException, Request, Header, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from datetime import datetime
import uuid
from typing import Optional

from src.core.config import config
from src.core.logging import logger
from src.core.client import OpenAIClient
from src.models.claude import ClaudeMessagesRequest, ClaudeTokenCountRequest
from src.conversion.request_converter import convert_claude_to_openai
from src.conversion.response_converter import (
    convert_openai_to_claude_response,
    convert_openai_streaming_to_claude_with_cancellation,
)
from src.core.model_manager import model_manager

router = APIRouter()

# Get custom headers from config
custom_headers = config.get_custom_headers()

openai_client = OpenAIClient(
    config.openai_api_key,
    config.openai_base_url,
    config.request_timeout,
    api_version=config.azure_api_version,
    custom_headers=custom_headers,
)

async def validate_api_key(x_api_key: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
    """Validate the client's API key from either x-api-key header or Authorization header."""
    client_api_key = None
    
    # Extract API key from headers
    if x_api_key:
        client_api_key = x_api_key
    elif authorization and authorization.startswith("Bearer "):
        client_api_key = authorization.replace("Bearer ", "")
    
    # Skip validation if ANTHROPIC_API_KEY is not set in the environment
    if not config.anthropic_api_key:
        return
        
    # Validate the client API key
    if not client_api_key or not config.validate_client_api_key(client_api_key):
        logger.warning(f"Invalid API key provided by client")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Please provide a valid Anthropic API key."
        )

@router.post("/v1/messages")
async def create_message(request: ClaudeMessagesRequest, http_request: Request, _: None = Depends(validate_api_key)):
    try:
        logger.debug(
            f"Processing Claude request: model={request.model}, stream={request.stream}"
        )

        # Generate unique request ID for cancellation tracking
        request_id = str(uuid.uuid4())

        # Convert Claude request to OpenAI format
        openai_request = convert_claude_to_openai(request, model_manager)

        # Check if client disconnected before processing
        if await http_request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        if request.stream:
            # Streaming response - wrap in error handling
            try:
                openai_stream = openai_client.create_chat_completion_stream(
                    openai_request, request_id
                )
                return StreamingResponse(
                    convert_openai_streaming_to_claude_with_cancellation(
                        openai_stream,
                        request,
                        logger,
                        http_request,
                        openai_client,
                        request_id,
                    ),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "*",
                    },
                )
            except HTTPException as e:
                # Convert to proper error response for streaming
                logger.error(f"Streaming error: {e.detail}")
                import traceback

                logger.error(traceback.format_exc())
                error_message = openai_client.classify_openai_error(e.detail)
                error_response = {
                    "type": "error",
                    "error": {"type": "api_error", "message": error_message},
                }
                return JSONResponse(status_code=e.status_code, content=error_response)
        else:
            # Non-streaming response
            openai_response = await openai_client.create_chat_completion(
                openai_request, request_id
            )
            claude_response = convert_openai_to_claude_response(
                openai_response, request
            )
            return claude_response
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        logger.error(f"Unexpected error processing request: {e}")
        logger.error(traceback.format_exc())
        error_message = openai_client.classify_openai_error(str(e))
        raise HTTPException(status_code=500, detail=error_message)


@router.post("/v1/messages/count_tokens")
async def count_tokens(request: ClaudeTokenCountRequest, _: None = Depends(validate_api_key)):
    try:
        # For token counting, we'll use a simple estimation
        # In a real implementation, you might want to use tiktoken or similar

        total_chars = 0

        # Count system message characters
        if request.system:
            if isinstance(request.system, str):
                total_chars += len(request.system)
            elif isinstance(request.system, list):
                for block in request.system:
                    if hasattr(block, "text"):
                        total_chars += len(block.text)

        # Count message characters
        for msg in request.messages:
            if msg.content is None:
                continue
            elif isinstance(msg.content, str):
                total_chars += len(msg.content)
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if hasattr(block, "text") and block.text is not None:
                        total_chars += len(block.text)

        # Rough estimation: 4 characters per token
        estimated_tokens = max(1, total_chars // 4)

        return {"input_tokens": estimated_tokens}

    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "openai_api_configured": bool(config.openai_api_key),
        "api_key_valid": config.validate_api_key(),
        "client_api_key_validation": bool(config.anthropic_api_key),
    }


@router.get("/test-connection")
async def test_connection():
    """Test API connectivity to OpenAI"""
    try:
        # Simple test request to verify API connectivity
        test_response = await openai_client.create_chat_completion(
            {
                "model": config.small_model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
            }
        )

        return {
            "status": "success",
            "message": "Successfully connected to OpenAI API",
            "model_used": config.small_model,
            "timestamp": datetime.now().isoformat(),
            "response_id": test_response.get("id", "unknown"),
        }

    except Exception as e:
        logger.error(f"API connectivity test failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "failed",
                "error_type": "API Error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
                "suggestions": [
                    "Check your OPENAI_API_KEY is valid",
                    "Verify your API key has the necessary permissions",
                    "Check if you have reached rate limits",
                ],
            },
        )


@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Claude-to-OpenAI API Proxy v1.0.0",
        "status": "running",
        "config": {
            "openai_base_url": config.openai_base_url,
            "max_tokens_limit": config.max_tokens_limit,
            "api_key_configured": bool(config.openai_api_key),
            "client_api_key_validation": bool(config.anthropic_api_key),
            "big_model": config.big_model,
            "small_model": config.small_model,
        },
        "endpoints": {
            "messages": "/v1/messages",
            "count_tokens": "/v1/messages/count_tokens",
            "health": "/health",
            "test_connection": "/test-connection",
        },
    }

@router.get("/api/hello")
@router.get("/v1/oauth/hello")
async def hello():
    """
    A tiny health‑check endpoint that returns a JSON payload
    indicating the service is running.  It is intentionally
    lightweight and does *not* require an API key.
    """
    return {"message": "hello"}

from fastapi import Response

# ------------------------------------------------------------------
#  HEAD /api/hello
# ------------------------------------------------------------------
@router.head("/api/hello")
async def hello_head():
    # Return an empty body – just the status/headers
    return Response(status_code=200)

# ------------------------------------------------------------------
#  HEAD /v1/oauth/hello
# ------------------------------------------------------------------
@router.head("/v1/oauth/hello")
async def hello_oauth_head():
    return Response(status_code=200)

import uuid
import secrets
import time
from urllib.parse import urlparse, urlunparse, urlencode, parse_qsl
from fastapi import FastAPI, HTTPException, Query, Request
from starlette.responses import RedirectResponse
from collections import defaultdict

clients = defaultdict(lambda: {'user': 'default'})

auth_codes = {}
tokens = {}

@router.get("/oauth/authorize")
async def authorize_endpoint(request: Request):
    # Extract query parameters
    client_id = request.query_params.get('client_id')
    redirect_uri = request.query_params.get('redirect_uri')
    state = request.query_params.get('state')

    # Generate auth code
    auth_code = secrets.token_urlsafe(16)

    # Store auth code mapping
    user = clients[client_id]['user']
    auth_codes[auth_code] = {"user": user, "expires_at": time.time() + 300, "redirect_uri": redirect_uri, "state": state}

    # Combine code and state as per user's requirement
    if not state:
        raise HTTPException(status_code=400, detail="State parameter is missing")
    
    combined_code = f"{auth_code}#{state}"
    
    return {"authorization_code": combined_code}

def _build_redirect_url(redirect_uri: str, code: str, state: str) -> str:
    # Parse the redirect URI
    parsed = urlparse(redirect_uri)
    # Parameters to be added to the fragment
    fragment_params = {'authorization_code': code}
    if state:
        fragment_params['state'] = state
    
    # Construct the fragment string
    fragment_string = urlencode(fragment_params)
    
    # Build new URL with the fragment
    new_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, '', fragment_string))
    return new_url

@router.get("/oauth/redirect")
async def redirect_endpoint(request: Request):
    # Extract client_id and authorization code
    client_id = request.query_params.get('client_id')
    if not client_id or client_id not in clients:
        return {"error": "Client ID is missing or invalid"}
    authorization_code = request.query_params.get('authorization_code')
    if not authorization_code:
        return {"error": "Authorization code is missing"}
    # Verify auth code
    if authorization_code not in auth_codes:
        return {"error": "Invalid authorization code"}
    # Check expiration
    if time.time() > auth_codes[authorization_code]["expires_at"]:
        del auth_codes[authorization_code]
        return {"error": "Authorization code has expired"}
    # Generate token
    token = secrets.token_urlsafe(16)
    user = auth_codes[authorization_code]["user"]
    # Store token
    tokens[token] = {"user": user, "expires_at": time.time() + 3600}
    # Build final redirect URL
    redirect_uri = auth_codes[authorization_code]["redirect_uri"]
    state = auth_codes[authorization_code]["state"]
    final_url = _build_redirect_url(redirect_uri, token, state)
    return RedirectResponse(url=final_url)

from jose import jwt

@router.post("/v1/oauth/token")
async def token_endpoint(client_id: str = Query(None), client_secret: str = Query(None), grant_type: str = Query(None), authorization_code: str = Query(None)):
    # For this mock server, we'll perform minimal validation.
    # In a real-world scenario, you'd have robust validation.
    if not all([client_id, client_secret, grant_type, authorization_code]):
        raise HTTPException(status_code=400, detail="Missing required parameters")

    # Check grant_type
    if grant_type != "authorization_code":
        raise HTTPException(status_code=400, detail="Unsupported grant type")

    # Retrieve user (mock user)
    user = clients[client_id]['user'] if client_id in clients else 'default_user'

    # Create JWT payload
    expires_in = 3600  # Token expires in 1 hour
    to_encode = {
        "sub": user,
        "iat": int(time.time()),
        "exp": int(time.time()) + expires_in,
        "jti": str(uuid.uuid4())  # Unique token identifier
    }

    # Encode the token
    encoded_jwt = jwt.encode(to_encode, config.jwt_secret_key, algorithm=config.jwt_algorithm)

    # Store token info if needed (optional, as JWT is self-contained)
    tokens[encoded_jwt] = {"user": user, "expires_at": to_encode['exp']}

    return {
        "access_token": encoded_jwt,
        "token_type": "bearer",
        "expires_in": expires_in,
    }

