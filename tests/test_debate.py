from graph import build_graph
from main import build_initial_state

def test():
    app = build_graph()
    initial_state = build_initial_state("Is AI good for society?")
    
    # We'll just run it with a limit so it only does 1-2 turns, or we capture the graph output stream.
    print("Running graph...")
    try:
        # The graph will block on check_human_interrupt which uses input().
        # Actually, let's just use app.stream to see each step.
        # But wait, input() inside check_human_interrupt will block.
        # Let's mock input.
        import builtins
        builtins.input = lambda prompt: "exit" # Always exit after first turn
        
        for event in app.stream(initial_state):
            print(event)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test()
