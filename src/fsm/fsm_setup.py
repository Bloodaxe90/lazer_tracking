from src.fsm.fsm import FSM


def setup_fsm(port: str, baudrate: int, timeout: int) -> FSM:
    fsm = FSM(port, baudrate, timeout)
    fsm.connect()
    fsm.send_command("system -R")
    fsm.send_command("system mems")
    fsm.send_command("signal input --source=waveform --axis=xy")
    return fsm