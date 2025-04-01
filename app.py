from flask import Flask, request, jsonify
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import uuid
import json

app = Flask(__name__)##code
from flask import Flask, request, jsonify
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import uuid
from flask_cors import CORS  # Import CORS
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# In-memory storage for negotiations
# Structure: {negotiation_id: {
#    "messages": [{"speaker": "user1", "message": "..."}, ...],
#    "user1_count": 0,
#    "user2_count": 0,
#    "total_count": 0
# }}
negotiations = {}

@app.route('/negotiate', methods=['POST'])
def negotiate():
    try:
        data = request.json
        
        # Validate incoming request
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Invalid request format"}), 400
        
        # Extract required fields
        negotiation_id = data.get('negotiation_id')
        speaker = data.get('speaker')
        message = data.get('message')
        
        # Validate fields
        if not speaker or not message:
            return jsonify({"error": "Missing required fields: 'speaker' and 'message' are required"}), 400
            
        if speaker not in ['user1', 'user2']:
            return jsonify({"error": "Speaker must be either 'user1' or 'user2'"}), 400
        
        # Create new negotiation if needed
        if not negotiation_id:
            negotiation_id = str(uuid.uuid4())
            negotiations[negotiation_id] = {
                "messages": [],
                "user1_count": 0,
                "user2_count": 0,
                "total_count": 0
            }
        elif negotiation_id not in negotiations:
            return jsonify({"error": f"Negotiation with ID {negotiation_id} not found"}), 404
        
        # Check if user has exceeded their 5-message limit
        if speaker == "user1" and negotiations[negotiation_id]["user1_count"] >= 5:
            return jsonify({"error": "User1 has already sent 5 messages"}), 400
        if speaker == "user2" and negotiations[negotiation_id]["user2_count"] >= 5:
            return jsonify({"error": "User2 has already sent 5 messages"}), 400
        
        # Add message to negotiation
        negotiations[negotiation_id]["messages"].append({
            "speaker": speaker,
            "message": message
        })
        
        # Update message counts
        if speaker == "user1":
            negotiations[negotiation_id]["user1_count"] += 1
        else:
            negotiations[negotiation_id]["user2_count"] += 1
            
        negotiations[negotiation_id]["total_count"] += 1
        
        # Check if we've reached 10 messages total (5 from each user)
        if negotiations[negotiation_id]["total_count"] == 10:
            # Generate verdict using LangChain and OpenAI
            verdict = generate_verdict(negotiations[negotiation_id]["messages"])
            
            # Clean up the negotiation data (optional)
            # del negotiations[negotiation_id]
            
            return jsonify({
                "negotiation_id": negotiation_id,
                "status": "completed",
                "verdict": verdict
            })
        else:
            # Return current status
            return jsonify({
                "negotiation_id": negotiation_id,
                "status": "in_progress",
                "messages_sent": negotiations[negotiation_id]["total_count"],
                "user1_messages": negotiations[negotiation_id]["user1_count"],
                "user2_messages": negotiations[negotiation_id]["user2_count"],
                "messages_remaining": 10 - negotiations[negotiation_id]["total_count"]
            })
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def generate_verdict(messages):
    """
    Generate a verdict using LangChain and OpenAI API based on the negotiation messages.
    
    Args:
        messages (list): List of dictionaries containing speaker and message content
    
    Returns:
        dict: Contains summary and compromise suggestion.
    """
    try:
        # Initialize the ChatOpenAI model
        chat = ChatOpenAI(
            model="gpt-3.5-turbo",  # Can use "gpt-3.5-turbo" for lower cost
            temperature=0.3,  # Lower temperature for more predictable responses
            api_key=os.environ.get("OPENAI_API_KEY"):
        )
        
        # Format the conversation for the AI
        conversation_text = "\n".join([f"{msg['speaker']}: {msg['message']}" for msg in messages])
        
        # Create system and user messages for the AI
        system_message = SystemMessage(content="""
        You are a fair and impartial legal mediator. Analyze the following negotiation between two parties 
        and provide a balanced verdict that represents a fair compromise. Your response must be in valid JSON 
        format with two fields:
        1. 'summary': A brief summary of the negotiation and the key points of contention
        2. 'compromise': A detailed middle-ground solution that addresses the concerns of both parties
        
        Respond only with the JSON object, no additional text or explanation.
        """)
        
        user_message = HumanMessage(content=f"Here is the negotiation conversation:\n\n{conversation_text}")
        
        # Get response from the AI
        response = chat([system_message, user_message])
        
        # Parse the JSON response
        try:
            verdict = json.loads(response.content)
            
            # Ensure response has required fields
            if "summary" not in verdict or "compromise" not in verdict:
                verdict = {
                    "summary": "Failed to generate proper verdict format.",
                    "compromise": "Please try again with clearer negotiation points."
                }
                
            return verdict
            
        except json.JSONDecodeError:
            # Handle case where AI doesn't return valid JSON
            return {
                "summary": "Error parsing AI response.",
                "compromise": "The negotiation was processed, but a structured verdict could not be generated. Please try again."
            }
            
    except Exception as e:
        print(f"Error generating verdict: {str(e)}")
        return {
            "summary": "Error generating verdict.",
            "compromise": f"An error occurred: {str(e)}"
        }


@app.route('/negotiation/<negotiation_id>', methods=['GET'])
def get_negotiation(negotiation_id):
    """
    Get the current status of a negotiation by ID, including the verdict if completed.
    """
    if negotiation_id not in negotiations:
        return jsonify({"error": "Negotiation not found"}), 404

    negotiation = negotiations[negotiation_id]
    
    response_data = {
        "negotiation_id": negotiation_id,
        "status": "in_progress" if negotiation["total_count"] < 10 else "completed",
        "messages_sent": negotiation["total_count"],
        "user1_messages": negotiation["user1_count"],
        "user2_messages": negotiation["user2_count"],
        "messages_remaining": max(0, 10 - negotiation["total_count"]),
        "messages": negotiation["messages"]
    }

    # If negotiation is completed, generate and include the verdict
    if negotiation["total_count"] == 10:
        verdict = generate_verdict(negotiation["messages"])
        response_data["verdict"] = verdict

    return jsonify(response_data)

if __name__ == '__main__':
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable is not set!")
        print("Set it before running the application: export OPENAI_API_KEY='your-api-key'")
    
    port = int(os.environ.get("PORT", 5500))
    app.run(host='0.0.0.0', port=port, debug=True)

# In-memory storage for negotiations
# Structure: {negotiation_id: {
#    "messages": [{"speaker": "user1", "message": "..."}, ...],
#    "user1_count": 0,
#    "user2_count": 0,
#    "total_count": 0
# }}
negotiations = {}

@app.route('/negotiate', methods=['POST'])
def negotiate():
    try:
        data = request.json
        
        # Validate incoming request
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Invalid request format"}), 400
        
        # Extract required fields
        negotiation_id = data.get('negotiation_id')
        speaker = data.get('speaker')
        message = data.get('message')
        
        # Validate fields
        if not speaker or not message:
            return jsonify({"error": "Missing required fields: 'speaker' and 'message' are required"}), 400
            
        if speaker not in ['user1', 'user2']:
            return jsonify({"error": "Speaker must be either 'user1' or 'user2'"}), 400
        
        # Create new negotiation if needed
        if not negotiation_id:
            negotiation_id = str(uuid.uuid4())
            negotiations[negotiation_id] = {
                "messages": [],
                "user1_count": 0,
                "user2_count": 0,
                "total_count": 0
            }
        elif negotiation_id not in negotiations:
            return jsonify({"error": f"Negotiation with ID {negotiation_id} not found"}), 404
        
        # Check if user has exceeded their 5-message limit
        if speaker == "user1" and negotiations[negotiation_id]["user1_count"] >= 5:
            return jsonify({"error": "User1 has already sent 5 messages"}), 400
        if speaker == "user2" and negotiations[negotiation_id]["user2_count"] >= 5:
            return jsonify({"error": "User2 has already sent 5 messages"}), 400
        
        # Add message to negotiation
        negotiations[negotiation_id]["messages"].append({
            "speaker": speaker,
            "message": message
        })
        
        # Update message counts
        if speaker == "user1":
            negotiations[negotiation_id]["user1_count"] += 1
        else:
            negotiations[negotiation_id]["user2_count"] += 1
            
        negotiations[negotiation_id]["total_count"] += 1
        
        # Check if we've reached 10 messages total (5 from each user)
        if negotiations[negotiation_id]["total_count"] == 10:
            # Generate verdict using LangChain and OpenAI
            verdict = generate_verdict(negotiations[negotiation_id]["messages"])
            
            # Clean up the negotiation data (optional)
            # del negotiations[negotiation_id]
            
            return jsonify({
                "negotiation_id": negotiation_id,
                "status": "completed",
                "verdict": verdict
            })
        else:
            # Return current status
            return jsonify({
                "negotiation_id": negotiation_id,
                "status": "in_progress",
                "messages_sent": negotiations[negotiation_id]["total_count"],
                "user1_messages": negotiations[negotiation_id]["user1_count"],
                "user2_messages": negotiations[negotiation_id]["user2_count"],
                "messages_remaining": 10 - negotiations[negotiation_id]["total_count"]
            })
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def generate_verdict(messages):
    """
    Generate a verdict using LangChain and OpenAI API based on the negotiation messages.
    
    Args:
        messages (list): List of dictionaries containing speaker and message content
    
    Returns:
        dict: Contains summary and compromise suggestion.
    """
    try:
        # Initialize the ChatOpenAI model
        chat = ChatOpenAI(
            model="gpt-3.5-turbo",  # Can use "gpt-3.5-turbo" for lower cost
            temperature=0.3,  # Lower temperature for more predictable responses
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        # Format the conversation for the AI
        conversation_text = "\n".join([f"{msg['speaker']}: {msg['message']}" for msg in messages])
        
        # Create system and user messages for the AI
        system_message = SystemMessage(content="""
        You are a fair and impartial legal mediator. Analyze the following negotiation between two parties 
        and provide a balanced verdict that represents a fair compromise. Your response must be in valid JSON 
        format with two fields:
        1. 'summary': A brief summary of the negotiation and the key points of contention
        2. 'compromise': A detailed middle-ground solution that addresses the concerns of both parties
        
        Respond only with the JSON object, no additional text or explanation.
        """)
        
        user_message = HumanMessage(content=f"Here is the negotiation conversation:\n\n{conversation_text}")
        
        # Get response from the AI
        response = chat([system_message, user_message])
        
        # Parse the JSON response
        try:
            verdict = json.loads(response.content)
            
            # Ensure response has required fields
            if "summary" not in verdict or "compromise" not in verdict:
                verdict = {
                    "summary": "Failed to generate proper verdict format.",
                    "compromise": "Please try again with clearer negotiation points."
                }
                
            return verdict
            
        except json.JSONDecodeError:
            # Handle case where AI doesn't return valid JSON
            return {
                "summary": "Error parsing AI response.",
                "compromise": "The negotiation was processed, but a structured verdict could not be generated. Please try again."
            }
            
    except Exception as e:
        print(f"Error generating verdict: {str(e)}")
        return {
            "summary": "Error generating verdict.",
            "compromise": f"An error occurred: {str(e)}"
        }

@app.route('/negotiation/<negotiation_id>', methods=['GET'])
def get_negotiation(negotiation_id):
    """
    Get the current status of a negotiation by ID.
    """
    if negotiation_id not in negotiations:
        return jsonify({"error": "Negotiation not found"}), 404
        
    return jsonify({
        "negotiation_id": negotiation_id,
        "status": "in_progress" if negotiations[negotiation_id]["total_count"] < 10 else "completed",
        "messages_sent": negotiations[negotiation_id]["total_count"],
        "user1_messages": negotiations[negotiation_id]["user1_count"],
        "user2_messages": negotiations[negotiation_id]["user2_count"],
        "messages_remaining": max(0, 10 - negotiations[negotiation_id]["total_count"]),
        "messages": negotiations[negotiation_id]["messages"]
    })

if __name__ == '__main__':
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable is not set!")
        print("Set it before running the application: export OPENAI_API_KEY='your-api-key'")
    
    port = int(os.environ.get("PORT", 5500))
    app.run(host='0.0.0.0', port=port, debug=True)
