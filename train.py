"""
Training script: trains the RL agent and saves checkpoints.
"""
if __name__ == "__main__":
    from source import System
    
    try:
        app = System()
        print("System initialized successfully")
        print("Starting training phase...")
        app.main()
        print("Training completed")
    except Exception as e:
        print(f"Fatal exception: {e}")
        import traceback
        traceback.print_exc()
    print("Application exited")
