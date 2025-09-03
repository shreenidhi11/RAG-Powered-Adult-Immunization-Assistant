import streamlit as st
import requests

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000/user_query"
FEEDBACK_URL = "http://127.0.0.1:8000/submit_feedback"

st.set_page_config(
    page_title="Recommended Adult Immunization Schedule QnA",
    page_icon="📖",
    layout="centered"
)

st.title("📖 Welcome to the Recommended Adult Immunization Schedule QnA")
st.write("Ask me anything about the Adult Immunization Schedule! (powered by FastAPI + RAG + Redis)")

# -----------------------------
# Functions
# -----------------------------
def save_feedback(index):
    """Save feedback from the widget into the message history."""
    print("from here")
    key = f"feedback_{index}"
    st.session_state.messages[index]["feedback"] = st.session_state.get(key)


def send_feedback(index, query, answer):
    """Send feedback from the widget to an API."""
    key = f"feedback_{index}"
    feedback_json = {
        "query": query,
        "answer": answer,
        "feedback_type": st.session_state.get(key)
    }
    print(feedback_json)
    requests.post(FEEDBACK_URL, json=feedback_json)
# -----------------------------
# Initialize session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Initialize session state variables
if "last_query" not in st.session_state:
    st.session_state.last_query = None

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None


# -----------------------------
# Display chat history
# -----------------------------
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Only assistant messages have feedback
        if msg["role"] == "assistant":
            prev_feedback = msg.get("feedback", None)
            st.session_state[f"feedback_{idx}"] = prev_feedback  # initialize widget state
            st.feedback(
                "thumbs",
                key=f"feedback_{idx}",
                disabled=prev_feedback is not None,
                on_change=save_feedback,
                args=[idx],
            )

# -----------------------------
# User input
# -----------------------------
if user_query := st.chat_input("Enter your question:"):
    # Append user message to history first
    st.session_state.messages.append({"role": "user", "content": user_query})
    user_index = len(st.session_state.messages) - 1

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)

    # Call backend
    with st.spinner("Thinking..."):
        try:
            response = requests.post(API_URL, json={"question": user_query})
            if response.status_code == 200:
                answer = response.json()["User Query"]

                # Store the query and answer in session state
                st.session_state.last_query = user_query
                st.session_state.last_answer = answer

                # Append assistant message first
                st.session_state.messages.append({"role": "assistant", "content": answer})
                assistant_index = len(st.session_state.messages) - 1

                # Display assistant message
                with st.chat_message("assistant"):
                    st.markdown(answer)

                # Add feedback widget for assistant
                st.feedback(
                    "thumbs",
                    key=f"feedback_{assistant_index}",
                    on_change=send_feedback,
                    args=[assistant_index, st.session_state.last_query, st.session_state.last_answer],
                )
                # this will always give you none because the user has not yet interacted with this widget
                #when it interacts the code is re-rendered and
                # print(st.session_state[f"feedback_{assistant_index}"])

            else:
                st.error(f"Error: {response.status_code}, {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")