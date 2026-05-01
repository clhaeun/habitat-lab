#!/usr/bin/env python3
"""
Headless habitat-sim viewer using OpenCV instead of pygame.
Avoids GLX conflicts on virtual displays.

Usage:
    env -u DISPLAY PYGAME_DISPLAY=:2 python examples/cv2_play.py --never-end

Controls (in the OpenCV window, click the window first to focus):
    i/k/j/l - Move base forward/back/left/right
    1-7     - Increase arm joint 0-6
    q-u     - Decrease arm joint 0-6
    o/p     - Snap/Unsnap gripper
    m       - Reset episode
    ESC     - Quit
"""

import argparse
import os

import cv2
import numpy as np

import habitat
import habitat.tasks.rearrange.rearrange_task  # noqa: F401
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import ThirdRGBSensorConfig
from habitat.utils.visualizations.utils import observations_to_image, overlay_frame

DEFAULT_CFG = "benchmark/rearrange/play/play.yaml"


def step_env(env, action_name, action_args):
    return env.step({"action": action_name, "action_args": action_args})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default=DEFAULT_CFG)
    parser.add_argument("--play-cam-res", type=int, default=512)
    parser.add_argument("--never-end", action="store_true", default=False)
    parser.add_argument("--save-obs", action="store_true", default=False)
    parser.add_argument("--save-obs-fname", type=str, default="play.mp4")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()

    config = habitat.get_config(args.cfg, args.opts)
    with habitat.config.read_write(config):
        env_config = config.habitat.environment
        sim_config = config.habitat.simulator
        task_config = config.habitat.task

        sim_config.debug_render = True
        agent_config = get_agent_config(sim_config=sim_config)
        agent_config.sim_sensors.update(
            {
                "third_rgb_sensor": ThirdRGBSensorConfig(
                    height=args.play_cam_res, width=args.play_cam_res
                )
            }
        )

        if "pddl_success" in task_config.measurements:
            task_config.measurements.pddl_success.must_call_stop = False
        if "rearrange_nav_to_obj_success" in task_config.measurements:
            task_config.measurements.rearrange_nav_to_obj_success.must_call_stop = False
        if "force_terminate" in task_config.measurements:
            task_config.measurements.force_terminate.max_accum_force = -1.0
            task_config.measurements.force_terminate.max_instant_force = -1.0

        if args.never_end:
            env_config.max_episode_steps = 0

    # Create env (habitat-sim uses EGL since DISPLAY is unset)
    with habitat.Env(config=config) as env:
        # Now restore DISPLAY for OpenCV window
        pygame_display = os.environ.get("PYGAME_DISPLAY")
        if pygame_display:
            os.environ["DISPLAY"] = pygame_display

        obs = env.reset()
        print("Environment created successfully!")
        print("Controls (click OpenCV window first):")
        print("  i/k/j/l  - Base forward/back/left/right")
        print("  1-7      - Increase arm joints 0-6")
        print("  q,w,e,r,t,y,u - Decrease arm joints 0-6")
        print("  o/p      - Snap/Unsnap gripper")
        print("  m        - Reset episode")
        print("  ESC      - Quit")

        # Determine action space info
        print("\n=== Action Space Debug ===")
        print(f"Action space keys: {list(env.action_space.spaces.keys())}")
        for k, v in env.action_space.spaces.items():
            if hasattr(v, 'spaces'):
                print(f"  {k}: {dict((sk, sv.shape) for sk, sv in v.spaces.items())}")
            else:
                print(f"  {k}: {v}")
        print("========================\n")

        multi_agent = len(env._sim.agents_mgr) > 1
        agent_k = "agent_0_" if multi_agent else ""
        arm_action_name = f"{agent_k}arm_action"
        base_action_name = f"{agent_k}base_velocity"

        has_arm = arm_action_name in env.action_space.spaces
        has_base = base_action_name in env.action_space.spaces
        print(f"multi_agent={multi_agent}, agent_k='{agent_k}'")
        print(f"has_arm={has_arm} ({arm_action_name}), has_base={has_base} ({base_action_name})")

        if has_arm:
            arm_space = env.action_space.spaces[arm_action_name]
            arm_dim = arm_space.spaces["arm_action"].shape[0]
        else:
            arm_dim = 0
        print(f"arm_dim={arm_dim}")

        # Key mappings for arm joint decrease: q,w,e,r,t,y,u
        decrease_keys = [ord("q"), ord("w"), ord("e"), ord("r"),
                         ord("t"), ord("y"), ord("u")]
        # Key mappings for arm joint increase: 1,2,3,4,5,6,7
        increase_keys = [ord("1"), ord("2"), ord("3"), ord("4"),
                         ord("5"), ord("6"), ord("7")]

        all_obs = []
        running = True
        while running:
            info = env.get_metrics()
            frame = observations_to_image(obs, info)
            frame = overlay_frame(frame, info)

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Habitat", frame_bgr)

            if args.save_obs:
                all_obs.append(frame)

            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # ESC
                running = False
                continue
            elif key == ord("m"):
                obs = env.reset()
                continue

            # Determine action based on key press
            action_name = "empty"
            action_args = {}

            # Base control
            if key == ord("i") and has_base:
                action_name = "base_velocity"
                action_args = {"base_vel": np.array([1, 0], dtype=np.float32)}
            elif key == ord("k") and has_base:
                action_name = "base_velocity"
                action_args = {"base_vel": np.array([-1, 0], dtype=np.float32)}
            elif key == ord("j") and has_base:
                action_name = "base_velocity"
                action_args = {"base_vel": np.array([0, 1], dtype=np.float32)}
            elif key == ord("l") and has_base:
                action_name = "base_velocity"
                action_args = {"base_vel": np.array([0, -1], dtype=np.float32)}
            # Gripper control
            elif key == ord("o") and has_arm:
                action_name = "arm_action"
                arm_action = np.zeros(arm_dim, dtype=np.float32)
                action_args = {"arm_action": arm_action, "grip_action": np.array([1.0], dtype=np.float32)}
            elif key == ord("p") and has_arm:
                action_name = "arm_action"
                arm_action = np.zeros(arm_dim, dtype=np.float32)
                action_args = {"arm_action": arm_action, "grip_action": np.array([-1.0], dtype=np.float32)}
            else:
                # Arm joint control
                for idx in range(min(7, arm_dim)):
                    if key == increase_keys[idx]:
                        action_name = "arm_action"
                        arm_action = np.zeros(arm_dim, dtype=np.float32)
                        arm_action[idx] = 1.0
                        action_args = {"arm_action": arm_action, "grip_action": np.zeros(1, dtype=np.float32)}
                        break
                    elif key == decrease_keys[idx]:
                        action_name = "arm_action"
                        arm_action = np.zeros(arm_dim, dtype=np.float32)
                        arm_action[idx] = -1.0
                        action_args = {"arm_action": arm_action, "grip_action": np.zeros(1, dtype=np.float32)}
                        break

            if key != 255:  # 255 = no key pressed
                print(f"Key: {chr(key) if 32 <= key < 127 else key}, action: {action_name}")

            obs = step_env(env, action_name, action_args)

        cv2.destroyAllWindows()

        if args.save_obs and all_obs:
            from habitat.utils.visualizations.utils import images_to_video
            os.makedirs("./data/vids", exist_ok=True)
            images_to_video(all_obs, "./data/vids", args.save_obs_fname.replace(".mp4", ""))
            print(f"Video saved to ./data/vids/{args.save_obs_fname}")


if __name__ == "__main__":
    main()
