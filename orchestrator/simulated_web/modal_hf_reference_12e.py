"""12e bounded automatic peer-note exposure; no automatic launch."""
from orchestrator.simulated_web.modal_hf_reference_12a import main
from orchestrator.simulated_web.peer_note_exposure import POLICY

if __name__ == "__main__":
    main(peer_note_exposure_policy=POLICY)
