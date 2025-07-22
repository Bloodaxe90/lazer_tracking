def wait(wait_message: str):
    while True:
        waiting = input(f"\n{wait_message} \nY/n ")

        if waiting.lower() == "y":
            break