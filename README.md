# SO-101 Studio

Browser SO-101 MuJoCo policy demo built with React, `mujoco-react`, LeRobot ACT,
and MolmoAct2 policy backends.

The default scene opens the MolmoAct2 SO-100/101 route:

- model: `allenai/MolmoAct2-SO100_101`
- policy endpoint: configure the MolmoAct2 inference URL in the UI
- browser app: `http://127.0.0.1:3001/`

MolmoAct2 is the default interactive route. The default Molmo scene is tuned for
a clean pickup: once the cube is lifted with a real gripper contact, the policy
pauses and holds the last policy controls. Continuing from pickup into placement
is still experimental and can be enabled with `taskAfterLift=...`. The ACT 12D
route below is the pick-place reference route:

- model: `davidlinjiahao/lerobot_so101_base_sim_pickplace`
- dataset/stats: `davidlinjiahao/lerobot_batch_001`
- policy endpoint: `http://127.0.0.1:8776/infer`
- browser app: `http://127.0.0.1:3001/?policy=act12`

Policy mode is intentionally pure: the remote policy is the only control writer.
There is no IK assist, scripted grasp, or object teleport path during rollout.

## Requirements

- Node.js 20+
- npm
- Python 3.12+
- `uv`
- Docker, optional, for the containerized policy backend

## Install

```bash
npm install
```

This project uses the published `mujoco-react` package from npm. It should not be
linked to a local library checkout.

## Run Locally

Start the browser app:

```bash
npm run dev
```

Start the policy server in another terminal:

```bash
npm run policy
```

Open `http://127.0.0.1:3001/`, choose a policy preset if needed, then press
`Run policy`.

The policy server uses `uv` and the `policy` dependency group in
`pyproject.toml`. To use a LeRobot checkout while developing the backend:

```bash
LEROBOT_ROOT=/path/to/lerobot npm run policy
```

## Verify

With the Vite app and policy server running:

```bash
npm run verify
```

The verifier opens the browser, starts the policy, waits for action chunks, and
checks policy responses, camera frames, contacts, cube lift, and final
cube-to-target distance. It writes:

- `artifacts/act12-verify.json`
- `artifacts/act12-verify.png`
- `artifacts/act12-frames/`

MolmoAct2 pickup verification uses the remote Molmo endpoint and the default
route:

```bash
node scripts/verify_policy_rollout.mjs \
  --url='http://127.0.0.1:3001/' \
  --inference-url='https://.../infer' \
  --duration-ms=75000 \
  --actions-per-request=45 \
  --min-responses=4 \
  --max-target-distance=99 \
  --min-cube-lift=0.07 \
  --min-final-cube-lift=0.06 \
  --min-target-contact-count=0 \
  --min-gripper-contact-count=100 \
  --min-moving-jaw-contact-count=60 \
  --object-body=red_cube \
  --target-body=green_target
```

## Policy Backend

The backend reads these environment variables:

- `MODEL_ID`, default `davidlinjiahao/lerobot_so101_base_sim_pickplace`
- `DATASET_ID`, default `davidlinjiahao/lerobot_batch_001`
- `POLICY_SPACE`, default `sim-radians-act`
- `POLICY_DEVICE`, default `auto`
- `POLICY_HOST`, default `127.0.0.1`
- `POLICY_PORT`, default `8776`

The health endpoint is `http://127.0.0.1:8776/health`.

## Docker Backend

Run the Python policy server in Docker:

```bash
docker compose -f compose.policy.yml up --build policy
```

The image defaults to CPU PyTorch wheels so it works on Apple Silicon and basic
cloud instances. For CUDA hosts, set the matching PyTorch wheel index and device:

```bash
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
POLICY_DEVICE=cuda \
docker compose -f compose.policy.yml up --build policy
```

The browser app can still run separately with `npm run dev`, or be built as
static assets with `npm run build`.

## `mujoco-react` Integration

This app is meant to show the React-facing policy primitives in real use:

- `MujocoProvider` and `MujocoCanvas` own the simulation lifecycle.
- `Debug` renders MuJoCo and virtual policy camera helpers.
- `ScenarioLighting` keeps viewer lighting separate from policy capture.
- `useRemotePolicy` schedules async inference requests and action chunks.
- `usePolicyCameraFramesFromMountedStreams` captures payload-ready policy images.
- `useObservation` reads simulation state for requests and telemetry.
- `useControlWriter` gates control ownership while policy mode is active.
- `applyPolicyActionToControls` writes policy actions with actuator clamping.
- `useBeforePhysicsStep` and `useAfterPhysicsStep` drive scene probes and HUDs.

The visible viewer can use smoother lighting and mesh rendering while the policy
camera uses isolated capture settings that match the policy's training view.

## Useful Query Params

Policy camera tuning:

- `policyCamPos=x,y,z`
- `policyCamLookAt=x,y,z`
- `policyCamUp=x,y,z`
- `policyCamFov=50`

Scene and physics tuning:

- `targetX`, `targetY`, `targetZ`
- `targetSX`, `targetSY`, `targetSZ`
- `goalX`, `goalY`, `goalZ`
- `goalRgba=r,g,b,a`
- `cubeMass`
- `cubeFriction`
- `cubeSolref`
- `cubeSolimp`
- `cubeCondim`
- `binWalls=false`
- `hideGoalInPolicyUntilLift=false`
- `taskAfterLiftLift=0.06`
- `autoPauseTicks=5`
- `taskAfterLift=put%20the%20red%20cube%20onto%20the%20green%20target`

Policy scheduling:

- `queue=replace` or `queue=append`
- `prefetch=4`

## Development

```bash
npm run typecheck
npm run build
docker compose -f compose.policy.yml config
```

Use `npm run verify` after any change that can affect policy observations,
scene physics, camera capture, or control application.
