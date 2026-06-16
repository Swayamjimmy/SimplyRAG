# Test multi-turn conversation with memory
from src.agent import stream_query

# First question on thread_1
stream_query("What method did the authors use?", thread_id="thread_1")

# Follow-up referencing previous answer (no need to re-specify topic)
stream_query("How does it compare to the baseline?", thread_id="thread_1")

# Different thread starts fresh (no memory from thread_1)
stream_query("How does it compare to the baseline?", thread_id="thread_2")