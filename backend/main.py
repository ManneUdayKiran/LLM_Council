"""FastAPI backend for LLM Council."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid
import json
import asyncio

try:
    # When running as a module (python -m uvicorn backend.main:app)
    from . import storage
    from .council import run_full_council, generate_conversation_title, stage1_collect_responses, stage2_collect_rankings, stage3_synthesize_final, calculate_aggregate_rankings
    from .prompt_normalizer import normalize_prompt
    from .semantic_consensus import analyze_semantic_consensus
    from .cost_router import route_query_with_cost_awareness
except ImportError:
    # When running directly from backend directory (uvicorn main:app)
    import storage
    from council import run_full_council, generate_conversation_title, stage1_collect_responses, stage2_collect_rankings, stage3_synthesize_final, calculate_aggregate_rankings
    from prompt_normalizer import normalize_prompt
    from semantic_consensus import analyze_semantic_consensus
    from cost_router import route_query_with_cost_awareness

app = FastAPI(title="LLM Council API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateConversationRequest(BaseModel):
    """Request to create a new conversation."""
    pass


class SendMessageRequest(BaseModel):
    """Request to send a message in a conversation."""
    content: str


class ConversationMetadata(BaseModel):
    """Conversation metadata for list view."""
    id: str
    created_at: str
    title: str
    message_count: int


class Conversation(BaseModel):
    """Full conversation with all messages."""
    id: str
    created_at: str
    title: str
    messages: List[Dict[str, Any]]


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "LLM Council API"}


@app.get("/api/conversations", response_model=List[ConversationMetadata])
async def list_conversations():
    """List all conversations (metadata only)."""
    return storage.list_conversations()


@app.delete("/api/conversations")
async def delete_all_conversations():
    """Delete all conversations."""
    count = storage.delete_all_conversations()
    return {"deleted": count}


@app.post("/api/conversations", response_model=Conversation)
async def create_conversation(request: CreateConversationRequest):
    """Create a new conversation."""
    conversation_id = str(uuid.uuid4())
    conversation = storage.create_conversation(conversation_id)
    return conversation


@app.get("/api/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str):
    """Get a specific conversation with all its messages."""
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.post("/api/conversations/{conversation_id}/message")
async def send_message(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and run the 3-stage council process.
    Returns the complete response with all stages.
    """
    # Check if conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Normalize the prompt
    normalized_content, norm_metadata = normalize_prompt(request.content)

    # Check if this is the first message
    is_first_message = len(conversation["messages"]) == 0

    # Add user message (store original, but process normalized)
    storage.add_user_message(conversation_id, request.content)

    # If this is the first message, generate a title
    if is_first_message:
        title = await generate_conversation_title(normalized_content)
        storage.update_conversation_title(conversation_id, title)

    # Run the 3-stage council process
    stage1_results, stage2_results, stage3_result, metadata = await run_full_council(
        normalized_content
    )
    
    # Add normalization metadata to response metadata
    metadata["prompt_normalization"] = norm_metadata
    
    # Analyze semantic consensus - convert stage1_results to dict format
    try:
        stage1_dict = {result["model"]: result["response"] for result in stage1_results}
        consensus_analysis = await analyze_semantic_consensus(
            stage1_dict,
            normalized_content,
            generate_merged=True
        )
        metadata["semantic_consensus"] = consensus_analysis
    except Exception as e:
        print(f"Semantic consensus error: {e}")
        metadata["semantic_consensus"] = {"error": str(e)}

    # Add assistant message with all stages
    storage.add_assistant_message(
        conversation_id,
        stage1_results,
        stage2_results,
        stage3_result
    )

    # Return the complete response with metadata
    return {
        "stage1": stage1_results,
        "stage2": stage2_results,
        "stage3": stage3_result,
        "metadata": metadata
    }


@app.post("/api/conversations/{conversation_id}/message/stream")
async def send_message_stream(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and stream the 3-stage council process.
    Returns Server-Sent Events as each stage completes.
    """
    # Check if conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Normalize the prompt
    normalized_content, norm_metadata = normalize_prompt(request.content)

    # Check if this is the first message
    is_first_message = len(conversation["messages"]) == 0

    async def event_generator():
        try:
            # Add user message (store original, but process normalized)
            storage.add_user_message(conversation_id, request.content)
            
            # Send normalization metadata
            yield f"data: {json.dumps({'type': 'normalization', 'metadata': norm_metadata})}\n\n"

            # Start title generation in parallel (don't await yet)
            title_task = None
            if is_first_message:
                title_task = asyncio.create_task(generate_conversation_title(normalized_content))

            # Stage 1: Collect responses
            yield f"data: {json.dumps({'type': 'stage1_start'})}\n\n"
            stage1_results = await stage1_collect_responses(normalized_content)
            yield f"data: {json.dumps({'type': 'stage1_complete', 'data': stage1_results})}\n\n"
            
            # Semantic Consensus Analysis - convert stage1_results to dict format
            try:
                yield f"data: {json.dumps({'type': 'consensus_start'})}\n\n"
                stage1_dict = {result["model"]: result["response"] for result in stage1_results}
                consensus_analysis = await analyze_semantic_consensus(
                    stage1_dict,
                    normalized_content,
                    generate_merged=True
                )
                yield f"data: {json.dumps({'type': 'consensus_complete', 'data': consensus_analysis})}\n\n"
            except Exception as e:
                print(f"Semantic consensus error: {e}")
                yield f"data: {json.dumps({'type': 'consensus_error', 'error': str(e)})}\n\n"

            # Stage 2: Collect rankings
            yield f"data: {json.dumps({'type': 'stage2_start'})}\n\n"
            stage2_results, label_to_model = await stage2_collect_rankings(normalized_content, stage1_results)
            aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)
            yield f"data: {json.dumps({'type': 'stage2_complete', 'data': stage2_results, 'metadata': {'label_to_model': label_to_model, 'aggregate_rankings': aggregate_rankings}})}\n\n"

            # Stage 3: Synthesize final answer
            yield f"data: {json.dumps({'type': 'stage3_start'})}\n\n"
            stage3_result = await stage3_synthesize_final(normalized_content, stage1_results, stage2_results)
            yield f"data: {json.dumps({'type': 'stage3_complete', 'data': stage3_result})}\n\n"

            # Wait for title generation if it was started
            if title_task:
                title = await title_task
                storage.update_conversation_title(conversation_id, title)
                yield f"data: {json.dumps({'type': 'title_complete', 'data': {'title': title}})}\n\n"

            # Save complete assistant message
            storage.add_assistant_message(
                conversation_id,
                stage1_results,
                stage2_results,
                stage3_result
            )

            # Send completion event
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        except Exception as e:
            # Send error event
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/api/conversations/{conversation_id}/cost-aware")
async def send_message_cost_aware(conversation_id: str, request: SendMessageRequest):
    """
    Send a message with cost-aware routing.
    Tries cheap models first, escalates to premium if needed.
    """
    # Check if conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Normalize the prompt
    normalized_content, norm_metadata = normalize_prompt(request.content)

    # Add user message
    storage.add_user_message(conversation_id, request.content)

    # Route with cost awareness
    routing_result = await route_query_with_cost_awareness(normalized_content)

    # Use premium responses if escalated, otherwise use cheap responses
    if routing_result["escalated"] and routing_result["premium_responses"]:
        final_consensus = routing_result["premium_consensus"]
        best_response = final_consensus.get("final_answer") or final_consensus.get("best_answer", {}).get("content", "")
    else:
        final_consensus = routing_result["cheap_consensus"]
        best_response = final_consensus.get("final_answer") or final_consensus.get("best_answer", {}).get("content", "")

    # Store the response
    storage.add_assistant_message(
        conversation_id,
        [],  # No stage1 in this mode
        [],  # No stage2 in this mode
        {"model": "cost-aware-router", "response": best_response}
    )

    return {
        "response": best_response,
        "routing_info": {
            "escalated": routing_result["escalated"],
            "total_cost": routing_result["total_cost"],
            "cost_breakdown": routing_result["cost_breakdown"],
            "routing_log": routing_result["routing_log"],
            "quality_report": routing_result["quality_report"],
        },
        "consensus": final_consensus,
        "metadata": {
            "prompt_normalization": norm_metadata,
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
