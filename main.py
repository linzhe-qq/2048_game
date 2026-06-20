import sys

from config import Config
from trainer import Trainer
from console_ui import ConsoleUI


def main():
    config = Config()
    trainer = Trainer(config)

    # Attempt checkpoint recovery
    recovered = False
    for tag in ("latest", "emergency", "manual"):
        try:
            if trainer.load_checkpoint(tag):
                recovered = True
                break
        except Exception:
            continue

    ui = ConsoleUI(trainer)

    if recovered:
        ep = trainer.current_episode
        step = trainer.total_steps
        print(f"[RECOVERY] Resuming from Episode {ep:,}, Step {step:,}")
        import time
        time.sleep(1.5)

    try:
        ui.run()
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        print("Check ./logs/error.log for details.", file=sys.stderr)
        trainer.save_checkpoint("emergency")
        sys.exit(1)


if __name__ == "__main__":
    main()
