import { useEffect, useMemo, useRef } from 'react';
import {
  applyPolicyActionToControls,
  findBodyByName,
  ModelActuators,
  useControlWriter,
  useMujoco,
  useObservation,
  usePolicyCameraFramesFromMountedStreams,
  useRemotePolicy,
} from 'mujoco-react';
import type {
  CreatePolicyCameraFrameCapturePlanOptions,
  MujocoData,
  MujocoModel,
  PolicyVector,
  RemotePolicyConfig,
  RemotePolicyResponseInfo,
} from 'mujoco-react';

export const ACT12_MODEL_ID = 'davidlinjiahao/lerobot_so101_base_sim_pickplace';
export const ACT12_SOURCE_REPO = 'davidlinjiahao/lerobot_batch_001';
export const ACT12_INFERENCE_URL = 'http://127.0.0.1:8776/infer';
export const ACT12_TASK = 'pick and place the cube';
export const ACT12_ROBOT_TYPE = 'so101';
export const ACT12_POLICY_FPS = 30;
export const ACT12_ACTIONS_PER_REQUEST = 100;
export const ACT12_POLICY_CAMERA = {
  position: [0.72, 0, 1.08],
  lookAt: [0.4, 0, 0.43],
  up: [0, 0, 1],
  fov: 50,
  width: 640,
  height: 480,
} as const;
export type LeRobotStateMode = 'cube-to-target-12';
export type LeRobotCameraPlanOptions = Omit<
  CreatePolicyCameraFrameCapturePlanOptions,
  'cameras' | 'sites' | 'bodies'
>;

const SO101_POLICY_ACTUATORS = [
  ModelActuators.so101.shoulder_pan,
  ModelActuators.so101.shoulder_lift,
  ModelActuators.so101.elbow_flex,
  ModelActuators.so101.wrist_flex,
  ModelActuators.so101.wrist_roll,
  ModelActuators.so101.gripper,
];

export type PolicyExecutionMode = 'raw-act';
export type PolicyQueueStrategy = NonNullable<RemotePolicyConfig['queueStrategy']>;

export interface PolicyTelemetry {
  sourceRepo: string;
  task: string;
  observationSize: number;
  state: number[];
  ctrl: number[];
  action: number[];
  modelAction?: number[];
  actionPolicyUnits?: number[];
  actionCount?: number;
  executionMode: PolicyExecutionMode;
  running: boolean;
  actionSource: string;
  cameraSource: string;
  policySpace?: string;
  stateDim?: number;
  statsSource?: string;
  captureMs?: number;
  inferenceMs?: number;
  requestMs?: number;
  queueStrategy?: PolicyQueueStrategy;
  prefetchThreshold?: number;
  remoteStatus?: string;
  requestCount?: number;
  responseCount?: number;
  queuedActions?: number;
  inFlight?: boolean;
}

interface LeRobotPolicyResponse {
  action?: number[];
  actions?: number[][];
  action_policy_units?: number[];
  policy_space?: string;
  state_dim?: number;
  stats_source?: string;
  model_id?: string;
  server_ms?: number;
  error?: string;
  [key: string]: unknown;
}

interface LeRobotDebugGlobal {
  __lerobotCameraFrames?: {
    at: number;
    [key: string]: string | number | undefined;
  };
  __lerobotCameraFrameHistory?: NonNullable<LeRobotDebugGlobal['__lerobotCameraFrames']>[];
  __lerobotPolicyDebug?: Array<{
    at: number;
    time: number;
    kind: 'request' | 'response' | 'error';
    state?: number[];
    ctrl?: number[];
    reset?: boolean;
    cameraSource?: string;
    firstAction?: number[];
    tenthAction?: number[];
    lastAction?: number[];
    actionCount?: number;
    captureMs?: number;
    inferenceMs?: number;
    requestMs?: number;
    serverMs?: number;
    responseMeta?: Record<string, unknown>;
    error?: string;
  }>;
}

interface UseLeRobotRemotePolicyOptions {
  enabled: boolean;
  executionMode: PolicyExecutionMode;
  actionsPerRequest?: number;
  inferenceUrl?: string;
  frequency?: number;
  queueStrategy?: PolicyQueueStrategy;
  prefetchThreshold?: number;
  task?: string;
  robotType?: string;
  stateMode?: LeRobotStateMode;
  cameraPlan?: LeRobotCameraPlanOptions;
  resetOnTaskChange?: boolean;
  onTelemetry?: (telemetry: PolicyTelemetry) => void;
}

const HOLD_ACTION = [
  -0.117329,
  -1.727294,
  1.69,
  -1.629534,
  -0.058387,
  -0.17453,
];

const FALLBACK_CUBE_POSITION = [0.4, 0.18, 0.432493] as const;
const FALLBACK_TARGET_POSITION = [0.4, -0.18, 0.42] as const;

function copyVector(input: PolicyVector) {
  return Array.from(input, (value) => Number(value));
}

function bodyPosition(model: MujocoModel, data: MujocoData, name: string, fallback: readonly [number, number, number]) {
  const bodyId = findBodyByName(model, name);
  if (bodyId < 0) return [...fallback];
  const index = bodyId * 3;
  return [
    data.xpos[index] ?? fallback[0],
    data.xpos[index + 1] ?? fallback[1],
    data.xpos[index + 2] ?? fallback[2],
  ];
}

function currentState(model: MujocoModel, data: MujocoData, stateMode: LeRobotStateMode) {
  const jointState = [
    data.qpos[0] ?? data.ctrl[0] ?? HOLD_ACTION[0],
    data.qpos[1] ?? data.ctrl[1] ?? HOLD_ACTION[1],
    data.qpos[2] ?? data.ctrl[2] ?? HOLD_ACTION[2],
    data.qpos[3] ?? data.ctrl[3] ?? HOLD_ACTION[3],
    data.qpos[4] ?? data.ctrl[4] ?? HOLD_ACTION[4],
    data.qpos[5] ?? data.ctrl[5] ?? HOLD_ACTION[5],
  ];

  const cube = bodyPosition(model, data, 'red_cube', FALLBACK_CUBE_POSITION);
  const target = bodyPosition(model, data, 'green_target', FALLBACK_TARGET_POSITION);
  const cubeToTarget = [
    target[0] - cube[0],
    target[1] - cube[1],
    target[2] - cube[2],
  ];

  switch (stateMode) {
    case 'cube-to-target-12':
      return [
        ...jointState,
        ...cube,
        ...cubeToTarget,
      ];
  }
}

function currentCtrl(data: MujocoData) {
  return Array.from(data.ctrl.slice(0, 6), (value) => Number(value));
}

function describeActionSource(mode: PolicyExecutionMode, hasModelAction: boolean) {
  void mode;
  return hasModelAction ? 'raw remote policy' : 'raw remote policy waiting; holding controls';
}

function appendPolicyDebug(entry: NonNullable<LeRobotDebugGlobal['__lerobotPolicyDebug']>[number]) {
  const global = globalThis as typeof globalThis & LeRobotDebugGlobal;
  const entries = global.__lerobotPolicyDebug ?? [];
  entries.push(entry);
  if (entries.length > 80) entries.splice(0, entries.length - 80);
  global.__lerobotPolicyDebug = entries;
}

function setPolicyDebugCameraFrames(frames: NonNullable<LeRobotDebugGlobal['__lerobotCameraFrames']>) {
  const global = globalThis as typeof globalThis & LeRobotDebugGlobal;
  global.__lerobotCameraFrames = frames;
  const history = global.__lerobotCameraFrameHistory ?? [];
  history.push(frames);
  if (history.length > 40) history.splice(0, history.length - 40);
  global.__lerobotCameraFrameHistory = history;
}

const CAPTURE_BACKGROUND = '#d8ddd8';
const CAPTURE_HIDDEN_GEOM_GROUPS = [3, 4] as const;
const CAPTURE_HIDDEN_FLOOR_GEOMS = ['floor', 'floor_box_geom'] as const;
const EMPTY_CAMERA_PLAN = {
  cameraKeys: [],
  requireAll: false,
} satisfies LeRobotCameraPlanOptions;

function parsePolicyCameraVectorParam(name: string): [number, number, number] | undefined {
  if (typeof window === 'undefined') return undefined;
  const value = new URLSearchParams(window.location.search).get(name);
  if (!value) return undefined;
  const parts = value.split(',').map((part) => Number(part.trim()));
  if (parts.length !== 3 || !parts.every(Number.isFinite)) return undefined;
  return [parts[0], parts[1], parts[2]];
}

function parsePolicyCameraNumberParam(name: string): number | undefined {
  if (typeof window === 'undefined') return undefined;
  const value = new URLSearchParams(window.location.search).get(name);
  if (!value) return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function cloneCameraPlanOptions(options: LeRobotCameraPlanOptions): LeRobotCameraPlanOptions {
  return {
    ...options,
    cameraKeys: [...options.cameraKeys],
    defaults: options.defaults ? { ...options.defaults } : undefined,
    streamOptions: options.streamOptions
      ? Object.fromEntries(
        Object.entries(options.streamOptions).map(([key, streamOptions]) => [
          key,
          { ...streamOptions },
        ]),
      )
      : undefined,
  };
}

function createDefaultLeRobotCameraPlanOptions(): LeRobotCameraPlanOptions {
  const defaults = {
    width: 640,
    type: 'image/jpeg',
    quality: 0.82,
    hiddenGeomGroups: CAPTURE_HIDDEN_GEOM_GROUPS,
    background: CAPTURE_BACKGROUND,
    renderIsolation: true,
    visualOverrides: {
      sceneEnvironment: null,
      sceneFog: null,
      shadows: false,
    },
    mujocoCameraCompatibility: {
      useResolution: true,
      useIntrinsics: true,
      useClipping: true,
      preserveAspect: true,
    },
  } satisfies LeRobotCameraPlanOptions['defaults'];

  const position = parsePolicyCameraVectorParam('policyCamPos') ?? [...ACT12_POLICY_CAMERA.position];
  const lookAt = parsePolicyCameraVectorParam('policyCamLookAt') ?? [...ACT12_POLICY_CAMERA.lookAt];
  const fov = parsePolicyCameraNumberParam('policyCamFov') ?? ACT12_POLICY_CAMERA.fov;
  return {
    cameraKeys: ['front'],
    requireAll: true,
    defaults: {
      ...defaults,
      fov,
      height: ACT12_POLICY_CAMERA.height,
      background: '#6c6c6c',
      hiddenGeomNames: CAPTURE_HIDDEN_FLOOR_GEOMS,
    },
    streamOptions: {
      front: {
        position,
        lookAt,
        up: [...ACT12_POLICY_CAMERA.up],
        fov,
      },
    },
  };
}

function createLeRobotCameraPlanOptions(basePlan?: LeRobotCameraPlanOptions): LeRobotCameraPlanOptions | null {
  const plan = cloneCameraPlanOptions(basePlan ?? createDefaultLeRobotCameraPlanOptions());
  const firstExplicitKey = plan.cameraKeys.find((key) => {
    const stream = plan.streamOptions?.[key];
    return Boolean(stream?.position || stream?.lookAt);
  });

  if (!firstExplicitKey) return plan;

  const position = parsePolicyCameraVectorParam('policyCamPos');
  const lookAt = parsePolicyCameraVectorParam('policyCamLookAt');
  const up = parsePolicyCameraVectorParam('policyCamUp');
  const fov = parsePolicyCameraNumberParam('policyCamFov');
  if (!position && !lookAt && !up && fov === undefined) return plan;

  const streamOptions = {
    ...(plan.streamOptions ?? {}),
    [firstExplicitKey]: {
      ...(plan.streamOptions?.[firstExplicitKey] ?? {}),
      ...(position ? { position } : {}),
      ...(lookAt ? { lookAt } : {}),
      ...(up ? { up } : {}),
      ...(fov === undefined ? {} : { fov }),
    },
  };

  return {
    ...plan,
    streamOptions,
  };
}

export function useLeRobotRemotePolicy({
  enabled,
  executionMode,
  actionsPerRequest = ACT12_ACTIONS_PER_REQUEST,
  inferenceUrl = ACT12_INFERENCE_URL,
  frequency = ACT12_POLICY_FPS,
  queueStrategy = 'replace',
  prefetchThreshold,
  task = ACT12_TASK,
  robotType = ACT12_ROBOT_TYPE,
  stateMode = 'cube-to-target-12',
  cameraPlan,
  resetOnTaskChange = true,
  onTelemetry,
}: UseLeRobotRemotePolicyOptions) {
  const mujoco = useMujoco();
  const observation = useObservation({
    qpos: true,
    qvel: true,
    ctrl: true,
    sites: ['gripperframe'],
  });
  const lastTelemetryTimeRef = useRef(-1);
  const serverStatusRef = useRef('waiting for policy server');
  const cameraSourceRef = useRef('waiting for scene cameras');
  const lastTimingRef = useRef<Pick<PolicyTelemetry, 'captureMs' | 'inferenceMs' | 'requestMs'>>({});
  const lastResponseMetaRef = useRef<Pick<
    PolicyTelemetry,
    'actionPolicyUnits' | 'actionCount' | 'policySpace' | 'stateDim' | 'statsSource'
  >>({});
  const controlWriter = useControlWriter({
    owner: 'remote-policy',
    selector: { actuators: SO101_POLICY_ACTUATORS },
    enabled,
  });
  const cameraPlanOptions = useMemo(
    () => createLeRobotCameraPlanOptions(cameraPlan) ?? EMPTY_CAMERA_PLAN,
    [cameraPlan]
  );
  const policyCameras = usePolicyCameraFramesFromMountedStreams(cameraPlanOptions);

  useEffect(() => {
    if (!enabled) return;
    serverStatusRef.current = 'waiting for policy server';
    cameraSourceRef.current = 'waiting for scene cameras';
    lastTimingRef.current = {};
  }, [
    enabled,
    executionMode,
    actionsPerRequest,
    inferenceUrl,
    resetOnTaskChange ? task : undefined,
    robotType,
    frequency,
    stateMode,
    queueStrategy,
    prefetchThreshold,
    cameraPlanOptions,
  ]);

  const resolvedPrefetchThreshold = prefetchThreshold
    ?? Math.max(1, Math.min(8, Math.floor(actionsPerRequest / 4)));

  const policy = useRemotePolicy({
    endpoint: inferenceUrl,
    frequency,
    enabled,
    queueStrategy,
    prefetchThreshold: resolvedPrefetchThreshold,
    clearQueueOnStop: true,
    onObservation: () => observation.readValues(),
    buildRequest: async ({ model, data, reset, signal, queuedActions }) => {
      const api = mujoco.api;
      const state = currentState(model, data, stateMode);
      const ctrl = currentCtrl(data);
      const simTime = data.time;
      signal.throwIfAborted();

      const usesCameras = cameraPlanOptions.cameraKeys.length > 0;
      if (!api && usesCameras) {
        cameraSourceRef.current = 'scene camera API unavailable';
        throw new Error('Scene camera API unavailable');
      }

      const requestStartedAt = performance.now();
      let captureFinishedAt = requestStartedAt;
      const frameBundle = usesCameras
        ? await policyCameras.capture()
        : null;
      signal.throwIfAborted();

      const images: Record<string, string> = frameBundle?.images ?? {};
      const preferredFrame = frameBundle
          ? frameBundle.frames.front
          ?? frameBundle.frames.top
          ?? frameBundle.frames.side
          ?? frameBundle.frames.wrist
          ?? Object.values(frameBundle.frames)[0]
        : undefined;
      if (frameBundle && preferredFrame) {
        cameraSourceRef.current = frameBundle.sourceSummary;
        setPolicyDebugCameraFrames({
          ...Object.fromEntries(
            Object.entries(frameBundle.frames).map(([key, frame]) => [key, frame.dataUrl])
          ),
          at: frameBundle.capturedAt,
        });
      } else {
        cameraSourceRef.current = 'not used by policy';
      }

      captureFinishedAt = performance.now();
      lastTimingRef.current = {
        captureMs: captureFinishedAt - requestStartedAt,
      };
      appendPolicyDebug({
        at: Date.now(),
        time: simTime,
        kind: 'request',
        state,
        ctrl,
        reset,
        cameraSource: cameraSourceRef.current,
        captureMs: lastTimingRef.current.captureMs,
      });

      return {
        state,
        time: simTime,
        reset,
        queued_actions: queuedActions ?? 0,
        images,
        task,
        robot_type: robotType,
      };
    },
    parseResponse: (responseBody: unknown, info: RemotePolicyResponseInfo) => {
      const result = responseBody as LeRobotPolicyResponse;
      if (!Array.isArray(result.action)) {
        throw new Error(result.error ?? 'No action returned');
      }
      const actions = (Array.isArray(result.actions) && result.actions.length > 0
        ? result.actions
        : [result.action])
        .slice(0, actionsPerRequest)
        .filter((action): action is number[] => Array.isArray(action))
        .map((action) => action.slice(0, 6));

      if (actions.length === 0) {
        throw new Error(result.error ?? 'No valid actions returned');
      }

      serverStatusRef.current = result.model_id ?? ACT12_MODEL_ID;
      lastResponseMetaRef.current = {
        actionPolicyUnits: Array.isArray(result.action_policy_units)
          ? result.action_policy_units.map(Number)
          : undefined,
        actionCount: actions.length,
        policySpace: result.policy_space,
        stateDim: result.state_dim,
        statsSource: result.stats_source,
      };
      const captureMs = lastTimingRef.current.captureMs ?? 0;
      lastTimingRef.current = {
        captureMs,
        inferenceMs: info.requestMs - captureMs,
        requestMs: info.requestMs,
      };
      appendPolicyDebug({
        at: Date.now(),
        time: info.data.time,
        kind: 'response',
        firstAction: actions[0],
        tenthAction: actions[9],
        lastAction: actions.at(-1),
        actionCount: actions.length,
        serverMs: result.server_ms,
        responseMeta: Object.fromEntries(
          Object.entries(result).filter(([key]) => ![
            'action',
            'actions',
            'model_id',
            'server_ms',
            'error',
          ].includes(key)),
        ),
        ...lastTimingRef.current,
      });

      return actions;
    },
    onError: (error) => {
      serverStatusRef.current = 'policy server unavailable';
      cameraSourceRef.current = mujoco.api ? 'scene cameras captured; server unavailable' : 'scene camera API unavailable';
      appendPolicyDebug({
        at: Date.now(),
        time: performance.now(),
        kind: 'error',
        error: error instanceof Error ? error.message : String(error),
      });
    },
    onAction: ({ action: modelPolicyAction, observation: policyObservation, model, data }) => {
      const modelAction = copyVector(modelPolicyAction);
      const ctrl = currentCtrl(data);
      const state = currentState(model, data, stateMode);
      const action = modelAction;

      if (data.time - lastTelemetryTimeRef.current > 0.18) {
        onTelemetry?.({
          sourceRepo: serverStatusRef.current,
          task,
          observationSize: policyObservation.length,
          state,
          ctrl,
          action,
          modelAction,
          executionMode,
          running: enabled,
          actionSource: describeActionSource(executionMode, true),
          cameraSource: cameraSourceRef.current,
          queueStrategy,
          prefetchThreshold: resolvedPrefetchThreshold,
          remoteStatus: policy.remoteStatus,
          requestCount: policy.requestCount,
          responseCount: policy.responseCount,
          queuedActions: policy.queuedActions,
          inFlight: policy.inFlight,
          ...lastResponseMetaRef.current,
          ...lastTimingRef.current,
        });
        lastTelemetryTimeRef.current = data.time;
      }
      if (!controlWriter.canWrite()) {
        appendPolicyDebug({
          at: Date.now(),
          time: data.time,
          kind: 'error',
          error: 'remote-policy control writer is not active',
        });
        return;
      }
      applyPolicyActionToControls(model, data, action);
    },
  });

  useEffect(() => {
    if (enabled) policy.start();
    else policy.stop();
  }, [enabled, policy]);

  useEffect(() => {
    policy.reset();
  }, [
    policy,
    actionsPerRequest,
    inferenceUrl,
    resetOnTaskChange ? task : undefined,
    robotType,
    frequency,
    stateMode,
    queueStrategy,
    resolvedPrefetchThreshold,
    resetOnTaskChange ? cameraPlanOptions : undefined,
  ]);

  return policy;
}

export function LeRobotPolicyRunner(options: UseLeRobotRemotePolicyOptions) {
  useLeRobotRemotePolicy(options);
  return null;
}
