import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from reach_env_cfg import ReachVlaEnvCfg

# Load Model (OpenVLA 7B)
model_id = "openvla/openvla-7b"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to("cuda:0")

prompt = "In: What action should the robot take to trim the palm flower? \nOut:"

# Initialize the environment with the VLA specific config
env_cfg = ReachVlaEnvCfg()
env = ManagerBasedRLEnv(cfg=env_cfg)
obs, _ = env.reset()

while simulation_app.is_running():
    # 1. Extract image from the sensor (Isaac Lab format is often [N, H, W, C])
    # The key depends on your ObservationsCfg definition
    raw_image = obs["policy"]["wrist_camera"][0].cpu().numpy() 
    image_pil = Image.fromarray(raw_image)

    # 2. VLA Inference
    inputs = processor(prompt, image_pil).to("cuda:0", dtype=torch.bfloat16)
    vla_action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

    # 3. Map to Robot Actions
    # Arm: [dx, dy, dz, droll, dpitch, dyaw]
    arm_cmd = torch.tensor(vla_action[:6], device=env.device)
    # Gripper: Normalize to your URDF limits
    gripper_val = vla_action[6] * 1.57 
    gripper_cmd = torch.tensor([gripper_val], device=env.device)

    # 4. Step Simulation
    actions = torch.cat([arm_cmd, gripper_cmd], dim=-1).unsqueeze(0)
    obs, rewards, terminations, truncations, extras = env.step(actions)