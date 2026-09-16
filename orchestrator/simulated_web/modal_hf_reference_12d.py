"""Prepared opt-in 12d reference ablation; no automatic launch."""
from orchestrator.simulated_web.modal_hf_reference_12a import main

if __name__ == "__main__":
    main(ablation_policy='union-discovery-v1')
