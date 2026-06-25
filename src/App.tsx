import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { Html, OrbitControls } from '@react-three/drei';
import {
  createRootRoute,
  createRoute,
  createRouter,
  Outlet,
  RouterProvider,
} from '@tanstack/react-router';
import { ChevronRight, Pause, Play } from 'lucide-react';
import {
  Debug,
  findBodyByName,
  findGeomByName,
  findSiteByName,
  getContact,
  getName,
  MujocoCanvas,
  MujocoProvider,
  ScenarioLighting,
  useAfterPhysicsStep,
  useBeforePhysicsStep,
  useMujoco,
} from 'mujoco-react';
import type {
  DebugVirtualCamera,
  MujocoData,
  MujocoModel,
  SceneConfig,
} from 'mujoco-react';
import { LeRobotPolicyRunner } from './controllers/LeRobotPickPlacePolicy';
import type {
  PolicyQueueStrategy,
  PolicyTelemetry,
} from './controllers/LeRobotPickPlacePolicy';
import {
  createSo101SceneObjects,
  selectSo101PolicyPreset,
} from './policies/so101PolicyPresets';
import type { So101PolicyCameraPlan, So101PolicyStateMode } from './policies/so101PolicyPresets';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import {
  Field,
  FieldContent,
  FieldGroup,
  FieldLabel,
  FieldTitle,
} from '@/components/ui/field';
import { Input } from '@/components/ui/input';
import { NativeSelect, NativeSelectOption } from '@/components/ui/native-select';
import { Separator } from '@/components/ui/separator';
import { Switch } from '@/components/ui/switch';

const SO101_JOINT_LABELS = [
  'shoulder_pan',
  'shoulder_lift',
  'elbow_flex',
  'wrist_flex',
  'wrist_roll',
  'gripper',
];

interface DebugState {
  time: number;
  qpos: number[];
  ctrl: number[];
  bodies: Record<string, [number, number, number]>;
  geoms: Record<string, [number, number, number]>;
  sites: Record<string, [number, number, number]>;
  contacts: Array<{
    geom1: string;
    geom2: string;
    body1: string;
    body2: string;
    pos: [number, number, number];
    dist: number;
  }>;
  ncon: number;
}

interface CameraDebugGlobal {
  __so101CaptureFrame?: (options: {
    cameraName?: 'realsense_d435i' | 'wrist_cam';
    position?: [number, number, number];
    lookAt?: [number, number, number];
    quaternion?: [number, number, number, number];
    up?: [number, number, number];
    fov?: number;
    width?: number;
    height?: number;
    type?: string;
    quality?: number;
    hiddenGeomGroups?: readonly number[];
    visibleGeomGroups?: readonly number[];
    hiddenGeomNames?: readonly string[];
    background?: string | number;
    backgroundAlpha?: number;
    flipX?: boolean;
    positionOffset?: [number, number, number];
    quaternionOffset?: [number, number, number, number];
  }) => Promise<{ dataUrl: string } | null>;
  __so101SetRobotState?: (qpos: number[], ctrl?: number[]) => void;
  __so101SetObjectPose?: (
    bodyName: string,
    position: [number, number, number],
    quaternion?: [number, number, number, number],
  ) => void;
  __so101DebugState?: DebugState;
  __so101ContactHistory?: Array<DebugState['contacts'][number] & { time: number }>;
  __so101ContactPairCounts?: Record<string, number>;
  __so101AutoPause?: {
    armed: boolean;
    triggered: boolean;
    lift: number;
    time: number;
    qpos: number[];
    ctrl: number[];
    cube: [number, number, number] | null;
    gripperContacts: number;
    movingJawContacts: number;
  };
  __so101TaskPhase?: {
    initialTask: string;
    activeTask: string;
    taskAfterLift: string | null;
    switched: boolean;
    lift: number;
    time: number;
  };
}

interface HudElements {
  qpos: HTMLElement[];
  degrees: HTMLElement[];
  ctrl: HTMLElement[];
  home: HTMLElement | null;
}

function numericSearchParam(name: string, fallback: number) {
  const value = new URLSearchParams(window.location.search).get(name);
  if (value === null) return fallback;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function optionalNumericSearchParam(name: string) {
  const value = new URLSearchParams(window.location.search).get(name);
  if (value === null) return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function booleanSearchParam(name: string, fallback: boolean) {
  const value = new URLSearchParams(window.location.search).get(name);
  if (value === null) return fallback;
  return !['0', 'false', 'off', 'no'].includes(value.toLowerCase());
}

function policyQueueStrategySearchParam(fallback: PolicyQueueStrategy): PolicyQueueStrategy {
  const value = new URLSearchParams(window.location.search).get('queue');
  return value === 'append' || value === 'replace' ? value : fallback;
}

const searchParams = new URLSearchParams(window.location.search);
const policyPreset = selectSo101PolicyPreset(searchParams.get('policy') ?? searchParams.get('preset'));
const sceneFile = policyPreset.sceneFile;
const homeJoints = policyPreset.homeJoints;
const policyQueueStrategy = policyQueueStrategySearchParam(policyPreset.queueStrategy ?? 'replace');
const policyPrefetchThreshold = optionalNumericSearchParam('prefetch') ?? policyPreset.prefetchThreshold;
const policyTask = searchParams.get('task') ?? policyPreset.task;
const taskAfterLiftParam = searchParams.get('taskAfterLift');
const defaultMolmoTaskAfterLift = 'put the red cube into the target bin';
const policyTaskAfterLift = taskAfterLiftParam === null
  ? policyPreset.id === 'molmo' ? defaultMolmoTaskAfterLift : ''
  : ['', '0', 'false', 'off', 'no'].includes(taskAfterLiftParam.toLowerCase())
    ? ''
    : taskAfterLiftParam;
const policyTaskAfterLiftThreshold = numericSearchParam('taskAfterLiftLift', policyPreset.id === 'molmo' ? 0.06 : 0.075);
const policyAutoPauseOnLift = booleanSearchParam('autoPause', false);
const policyAutoPauseLiftThreshold = numericSearchParam('autoPauseLift', policyPreset.id === 'molmo' ? 0.09 : 0.075);
const policyAutoPauseStableTicks = numericSearchParam('autoPauseTicks', policyPreset.id === 'molmo' ? 5 : 3);
const showBinInPolicyAfterLift = booleanSearchParam('showBinInPolicyAfterLift', true);
const resetSceneBeforePolicy = booleanSearchParam('resetScene', policyPreset.id !== 'act12');

function vectorSearchParam(
  name: string,
  fallback: readonly [number, number, number],
): [number, number, number] {
  const value = searchParams.get(name);
  if (!value) return [...fallback];
  const parts = value.split(',').map((part) => Number(part.trim()));
  if (parts.length !== 3 || !parts.every(Number.isFinite)) return [...fallback];
  return [parts[0], parts[1], parts[2]];
}

function vectorTuple(value: unknown): [number, number, number] | null {
  if (!Array.isArray(value) || value.length !== 3) return null;
  const tuple = value.map(Number);
  if (!tuple.every(Number.isFinite)) return null;
  return [tuple[0], tuple[1], tuple[2]];
}

function createPolicyDebugCameras(cameraPlan: So101PolicyCameraPlan): DebugVirtualCamera[] {
  return cameraPlan.cameraKeys.flatMap((key, index) => {
    const stream = cameraPlan.streamOptions?.[key];
    const position = vectorTuple(stream?.position);
    const lookAt = vectorTuple(stream?.lookAt);
    if (!position || !lookAt) return [];
    const up = vectorTuple(stream?.up) ?? [0, 0, 1];
    const fov = typeof stream?.fov === 'number'
      ? stream.fov
      : typeof cameraPlan.defaults?.fov === 'number'
        ? cameraPlan.defaults.fov
        : 50;
    const width = typeof stream?.width === 'number'
      ? stream.width
      : typeof cameraPlan.defaults?.width === 'number'
        ? cameraPlan.defaults.width
        : 640;
    const height = typeof stream?.height === 'number'
      ? stream.height
      : typeof cameraPlan.defaults?.height === 'number'
        ? cameraPlan.defaults.height
        : 480;
    return [{
      name: `policy ${key}`,
      position: index === 0 ? vectorSearchParam('policyCamPos', position) : position,
      lookAt: index === 0 ? vectorSearchParam('policyCamLookAt', lookAt) : lookAt,
      up: index === 0 ? vectorSearchParam('policyCamUp', up) : up,
      fov: index === 0 ? numericSearchParam('policyCamFov', fov) : fov,
      width,
      height,
    }];
  });
}

const policyDebugCameras = createPolicyDebugCameras(policyPreset.policyCamera);
const targetBinGeomNames = new Set([
  'target_bin_left_wall_geom',
  'target_bin_right_wall_geom',
  'target_bin_front_wall_geom',
  'target_bin_back_wall_geom',
]);

function revealBinInPolicyCameraPlan(cameraPlan: So101PolicyCameraPlan): So101PolicyCameraPlan {
  const hiddenGeomNames = cameraPlan.defaults?.hiddenGeomNames?.filter((name) => (
    !targetBinGeomNames.has(name)
  ));
  return {
    ...cameraPlan,
    defaults: cameraPlan.defaults
      ? {
        ...cameraPlan.defaults,
        hiddenGeomNames,
      }
      : undefined,
  };
}

const redCubePosition = [
  numericSearchParam('targetX', policyPreset.redCubePosition[0]),
  numericSearchParam('targetY', policyPreset.redCubePosition[1]),
  numericSearchParam('targetZ', policyPreset.redCubePosition[2]),
] as [number, number, number];

const redCubeSize = [
  numericSearchParam('targetSX', policyPreset.redCubeSize[0]),
  numericSearchParam('targetSY', policyPreset.redCubeSize[1]),
  numericSearchParam('targetSZ', policyPreset.redCubeSize[2]),
] as [number, number, number];

const greenTargetPosition = [
  numericSearchParam('goalX', policyPreset.greenTargetPosition[0]),
  numericSearchParam('goalY', policyPreset.greenTargetPosition[1]),
  numericSearchParam('goalZ', policyPreset.greenTargetPosition[2]),
] as [number, number, number];

const redCubeFriction = searchParams.get('cubeFriction') ?? undefined;
const redCubeSolref = searchParams.get('cubeSolref') ?? undefined;
const redCubeSolimp = searchParams.get('cubeSolimp') ?? undefined;
const includeAct12BinWalls = searchParams.has('binWalls')
  ? searchParams.get('binWalls') !== 'false'
  : true;

const sceneConfig: SceneConfig = {
  src: '/models/so101/',
  sceneFile,
  homeJoints,
  xmlPatches: policyPreset.xmlPatches,
  sceneObjects: createSo101SceneObjects(policyPreset, redCubePosition, redCubeSize, greenTargetPosition, {
    redCubeMass: optionalNumericSearchParam('cubeMass'),
    redCubeFriction,
    redCubeSolref,
    redCubeSolimp,
    redCubeCondim: optionalNumericSearchParam('cubeCondim'),
    includeAct12BinWalls,
  }),
};

function LoadingOverlay() {
  const sim = useMujoco();
  if (sim.isReady) return null;
  return (
    <Html center>
      <Card
        className="rounded-lg border-slate-200/15 bg-slate-950/65 px-3 py-2 text-sm text-slate-300 shadow-2xl backdrop-blur-xl"
        size="sm"
      >
        {sim.isError ? sim.error : 'Loading SO-101...'}
      </Card>
    </Html>
  );
}

function getHudElements(): HudElements {
  return {
    qpos: SO101_JOINT_LABELS.map((label) => (
      document.querySelector<HTMLElement>(`[data-joint-qpos="${label}"]`)
    )).filter(Boolean) as HTMLElement[],
    degrees: SO101_JOINT_LABELS.map((label) => (
      document.querySelector<HTMLElement>(`[data-joint-deg="${label}"]`)
    )).filter(Boolean) as HTMLElement[],
    ctrl: SO101_JOINT_LABELS.map((label) => (
      document.querySelector<HTMLElement>(`[data-joint-ctrl="${label}"]`)
    )).filter(Boolean) as HTMLElement[],
    home: document.querySelector<HTMLElement>('[data-home-joints]'),
  };
}

function hasHudElements(elements: HudElements) {
  return (
    elements.qpos.length === SO101_JOINT_LABELS.length &&
    elements.degrees.length === SO101_JOINT_LABELS.length &&
    elements.ctrl.length === SO101_JOINT_LABELS.length &&
    elements.home !== null
  );
}

function updateJointHud(elements: HudElements, qpos: number[], ctrl: number[]) {
  for (let index = 0; index < SO101_JOINT_LABELS.length; index += 1) {
    const q = qpos[index] ?? 0;
    const c = ctrl[index] ?? 0;
    elements.qpos[index].textContent = `q ${q.toFixed(5)}`;
    elements.degrees[index].textContent = `${(q * 180 / Math.PI).toFixed(2)} deg`;
    elements.ctrl[index].textContent = `ctrl ${c.toFixed(5)}`;
  }
  if (elements.home) {
    elements.home.textContent = `homeJoints: [${qpos.map((value) => value.toFixed(6)).join(', ')}]`;
  }
}

function setHudText(selector: string, value: string) {
  const element = document.querySelector<HTMLElement>(selector);
  if (element) element.textContent = value;
}

function contactPairKey(bodyA: string, bodyB: string) {
  return [bodyA, bodyB].sort().join('::');
}

function updatePolicyHud(telemetry: PolicyTelemetry) {
  const modelGripper = telemetry.modelAction?.[5];
  const policyGripper = telemetry.actionPolicyUnits?.[5];
  const actionGripper = telemetry.action[5] ?? 0;

  setHudText('[data-policy-source]', `Action source: ${telemetry.actionSource}`);
  setHudText('[data-policy-task]', `Task: ${telemetry.task}`);
  setHudText('[data-policy-execution]', `Execution: ${telemetry.executionMode}`);
  setHudText('[data-policy-model]', `Model: ${telemetry.sourceRepo}`);
  setHudText(
    '[data-policy-remote]',
    `Remote: ${telemetry.remoteStatus ?? '-'} | req ${telemetry.requestCount ?? 0} / resp ${telemetry.responseCount ?? 0} | queue ${telemetry.queuedActions ?? 0}${telemetry.inFlight ? ' | in flight' : ''}`,
  );
  setHudText(
    '[data-policy-space]',
    `Policy space: ${telemetry.policySpace ?? '-'} | state dim ${telemetry.stateDim ?? telemetry.observationSize}`,
  );
  setHudText(
    '[data-policy-stats]',
    `Stats: ${telemetry.statsSource ?? '-'} | chunk ${telemetry.actionCount ?? 0}`,
  );
  setHudText('[data-policy-camera]', `Cameras: ${telemetry.cameraSource}`);
  setHudText('[data-policy-observation]', `Observation: ${telemetry.observationSize} values`);
  setHudText(
    '[data-policy-timing]',
    `Timing: capture ${telemetry.captureMs?.toFixed(0) ?? '-'} ms | infer ${telemetry.inferenceMs?.toFixed(0) ?? '-'} ms | total ${telemetry.requestMs?.toFixed(0) ?? '-'} ms`,
  );
  setHudText(
    '[data-policy-scheduler]',
    `Scheduler: ${telemetry.queueStrategy ?? '-'} | prefetch ${telemetry.prefetchThreshold ?? '-'}`,
  );
  setHudText(
    '[data-policy-gripper]',
    modelGripper === undefined
      ? `Gripper: waiting | applied ${(actionGripper * 180 / Math.PI).toFixed(2)} deg`
      : `Gripper: raw ${policyGripper?.toFixed(2) ?? '-'} | sim ${modelGripper.toFixed(4)} rad (${(modelGripper * 180 / Math.PI).toFixed(2)} deg)`,
  );
  setHudText('[data-policy-action]', `Action: [${telemetry.action.map((value) => value.toFixed(4)).join(', ')}]`);
  setHudText(
    '[data-policy-action-units]',
    `Policy units: [${(telemetry.actionPolicyUnits ?? []).map((value) => value.toFixed(3)).join(', ')}]`,
  );
  setHudText(
    '[data-policy-model-action]',
    `Sim action: [${(telemetry.modelAction ?? []).map((value) => value.toFixed(4)).join(', ')}]`,
  );
}

function bodyPosition(model: MujocoModel, data: MujocoData, name: string) {
  const bodyId = findBodyByName(model, name);
  if (bodyId < 0) return null;
  const index = bodyId * 3;
  return [
    data.xpos[index] ?? 0,
    data.xpos[index + 1] ?? 0,
    data.xpos[index + 2] ?? 0,
  ] as [number, number, number];
}

function geomPosition(model: MujocoModel, data: MujocoData, name: string) {
  const geomId = findGeomByName(model, name);
  if (geomId < 0) return null;
  const index = geomId * 3;
  return [
    data.geom_xpos[index] ?? 0,
    data.geom_xpos[index + 1] ?? 0,
    data.geom_xpos[index + 2] ?? 0,
  ] as [number, number, number];
}

function sitePosition(model: MujocoModel, data: MujocoData, name: string) {
  const siteId = findSiteByName(model, name);
  if (siteId < 0) return null;
  const index = siteId * 3;
  return [
    data.site_xpos[index] ?? 0,
    data.site_xpos[index + 1] ?? 0,
    data.site_xpos[index + 2] ?? 0,
  ] as [number, number, number];
}

function SetupDebugBridge({ policyRunning }: { policyRunning: boolean }) {
  const sim = useMujoco();
  const pendingRobotStateRef = useRef<{ qpos: number[]; ctrl?: number[] } | null>(null);
  const pendingObjectPoseRef = useRef<Array<{
    bodyName: string;
    position: [number, number, number];
    quaternion: [number, number, number, number];
  }>>([]);

  useEffect(() => {
    if (!policyRunning) return;
    pendingRobotStateRef.current = null;
    pendingObjectPoseRef.current = [];
    const debugGlobal = globalThis as typeof globalThis & CameraDebugGlobal;
    debugGlobal.__so101ContactHistory = [];
    debugGlobal.__so101ContactPairCounts = {};
  }, [policyRunning]);

  useEffect(() => {
    const debugGlobal = globalThis as typeof globalThis & CameraDebugGlobal;
    debugGlobal.__so101CaptureFrame = (options) => (
      sim.api?.captureCameraFrame({
        width: 640,
        height: 480,
        type: 'image/jpeg',
        quality: 0.82,
        ...options,
      }) ?? Promise.resolve(null)
    );
    debugGlobal.__so101SetRobotState = (qpos, ctrl) => {
      if (policyRunning) return;
      debugGlobal.__so101ContactHistory = [];
      debugGlobal.__so101ContactPairCounts = {};
      pendingRobotStateRef.current = { qpos, ctrl };
    };
    debugGlobal.__so101SetObjectPose = (bodyName, position, quaternion = [1, 0, 0, 0]) => {
      if (policyRunning) return;
      debugGlobal.__so101ContactHistory = [];
      debugGlobal.__so101ContactPairCounts = {};
      pendingObjectPoseRef.current.push({ bodyName, position, quaternion });
    };
    return () => {
      delete debugGlobal.__so101CaptureFrame;
      delete debugGlobal.__so101SetRobotState;
      delete debugGlobal.__so101SetObjectPose;
    };
  }, [policyRunning, sim.api]);

  useBeforePhysicsStep(({ model, data }) => {
    if (policyRunning) return;

    const pendingRobotState = pendingRobotStateRef.current;
    if (pendingRobotState) {
      pendingRobotStateRef.current = null;
      for (let i = 0; i < Math.min(6, pendingRobotState.qpos.length, data.qpos.length); i += 1) {
        data.qpos[i] = pendingRobotState.qpos[i];
      }
      for (let i = 0; i < Math.min(6, data.qvel.length); i += 1) {
        data.qvel[i] = 0;
      }
      const ctrl = pendingRobotState.ctrl ?? pendingRobotState.qpos;
      for (let i = 0; i < Math.min(6, ctrl.length, data.ctrl.length); i += 1) {
        data.ctrl[i] = ctrl[i];
      }
    }

    const objectPoses = pendingObjectPoseRef.current.splice(0);
    for (const objectPose of objectPoses) {
      const bodyId = findBodyByName(model, objectPose.bodyName);
      const jointStart = bodyId >= 0 ? model.body_jntadr[bodyId] : -1;
      const jointCount = bodyId >= 0 ? model.body_jntnum[bodyId] : 0;
      for (let jointOffset = 0; jointOffset < jointCount; jointOffset += 1) {
        const jointId = jointStart + jointOffset;
        if (model.jnt_type[jointId] !== 0) continue;
        const qposAdr = model.jnt_qposadr[jointId];
        data.qpos[qposAdr] = objectPose.position[0];
        data.qpos[qposAdr + 1] = objectPose.position[1];
        data.qpos[qposAdr + 2] = objectPose.position[2];
        data.qpos[qposAdr + 3] = objectPose.quaternion[0];
        data.qpos[qposAdr + 4] = objectPose.quaternion[1];
        data.qpos[qposAdr + 5] = objectPose.quaternion[2];
        data.qpos[qposAdr + 6] = objectPose.quaternion[3];
        const dofAdr = model.jnt_dofadr[jointId];
        for (let i = 0; i < 6; i += 1) {
          data.qvel[dofAdr + i] = 0;
        }
      }
    }
  });

  return null;
}

function SceneProbe() {
  const hudElementsRef = useRef<HudElements | null>(null);
  const lastJointUpdateTime = useRef(-Infinity);
  const lastSceneUpdateTime = useRef(-Infinity);

  useAfterPhysicsStep(({ model, data }) => {
    if (data.time - lastJointUpdateTime.current >= 0.1) {
      lastJointUpdateTime.current = data.time;
      if (!hudElementsRef.current || !hasHudElements(hudElementsRef.current)) {
        hudElementsRef.current = getHudElements();
      }
      if (hasHudElements(hudElementsRef.current)) {
        updateJointHud(
          hudElementsRef.current,
          Array.from(data.qpos.slice(0, 6)),
          Array.from(data.ctrl.slice(0, 6)),
        );
      }
    }

    if (data.time - lastSceneUpdateTime.current < 0.05) return;
    lastSceneUpdateTime.current = data.time;

    const contacts: DebugState['contacts'] = [];
    if ((data.ncon ?? 0) > 0) {
      const contactArray = data.contact;
      try {
        for (let i = 0; i < Math.min(data.ncon ?? 0, 40); i += 1) {
          const contact = getContact(contactArray, i);
          if (!contact) continue;
          const body1 = model.geom_bodyid[contact.geom1] ?? -1;
          const body2 = model.geom_bodyid[contact.geom2] ?? -1;
          contacts.push({
            geom1: getName(model, model.name_geomadr[contact.geom1]) || `geom_${contact.geom1}`,
            geom2: getName(model, model.name_geomadr[contact.geom2]) || `geom_${contact.geom2}`,
            body1: body1 >= 0 ? getName(model, model.name_bodyadr[body1]) : '',
            body2: body2 >= 0 ? getName(model, model.name_bodyadr[body2]) : '',
            pos: [
              contact.pos[0] ?? 0,
              contact.pos[1] ?? 0,
              contact.pos[2] ?? 0,
            ],
            dist: contact.dist,
          });
        }
      } finally {
        contactArray.delete?.();
      }
    }

    const nonFloorContacts = contacts.filter(
      (contact) => contact.geom1 !== 'floor' && contact.geom2 !== 'floor',
    );
    if (nonFloorContacts.length > 0) {
      const debugGlobal = globalThis as typeof globalThis & CameraDebugGlobal;
      const history = debugGlobal.__so101ContactHistory ?? [];
      const contactPairCounts = debugGlobal.__so101ContactPairCounts ?? {};
      for (const contact of nonFloorContacts) {
        history.push({ ...contact, time: data.time });
        const key = contactPairKey(contact.body1, contact.body2);
        contactPairCounts[key] = (contactPairCounts[key] ?? 0) + 1;
      }
      if (history.length > 2000) history.splice(0, history.length - 2000);
      debugGlobal.__so101ContactHistory = history;
      debugGlobal.__so101ContactPairCounts = contactPairCounts;
    }

    const bodies: DebugState['bodies'] = {};
    for (const name of ['cube', 'box', 'box2', 'box3', 'red_cube', 'blue_cube', 'green_target', 'table', 'gripper', 'moving_jaw_so101_v1']) {
      const position = bodyPosition(model, data, name);
      if (position) {
        bodies[name] = position;
        setHudText(`[data-object-position="${name}"]`, `${name}: [${position.map((value) => value.toFixed(3)).join(', ')}]`);
      }
    }

    const geoms: DebugState['geoms'] = {};
    for (const name of [
      'fixed_jaw_box1',
      'fixed_jaw_box6',
      'fixed_jaw_box7',
      'moving_jaw_box1',
      'moving_jaw_box2',
      'moving_jaw_box3',
      'cube_geom',
      'box_geom',
      'box2_geom',
    ]) {
      const position = geomPosition(model, data, name);
      if (position) geoms[name] = position;
    }

    const sites: DebugState['sites'] = {};
    const gripperframe = sitePosition(model, data, 'gripperframe');
    if (gripperframe) {
      sites.gripperframe = gripperframe;
      setHudText(
        '[data-site-position="gripperframe"]',
        `gripperframe: [${gripperframe.map((value) => value.toFixed(3)).join(', ')}]`,
      );
    }
    setHudText('[data-contact-count]', `Contacts: ${data.ncon ?? 0}`);

    (globalThis as typeof globalThis & CameraDebugGlobal).__so101DebugState = {
      time: data.time,
      qpos: Array.from(data.qpos.slice(0, 6)),
      ctrl: Array.from(data.ctrl.slice(0, 6)),
      bodies,
      geoms,
      sites,
      contacts,
      ncon: data.ncon ?? 0,
    };
  });

  return null;
}

function SceneChildren({
  policyRunning,
  heldPolicyCtrl,
  showPolicyCameraDebug,
  showMujocoCameraDebug,
  inferenceUrl,
  actionsPerRequest,
  queueStrategy,
  prefetchThreshold,
  frequency,
  task,
  robotType,
  stateMode,
  cameraPlan,
  resetOnTaskChange,
  clearQueueOnTaskChange,
  onPolicyTelemetry,
}: {
  policyRunning: boolean;
  heldPolicyCtrl: number[] | null;
  showPolicyCameraDebug: boolean;
  showMujocoCameraDebug: boolean;
  inferenceUrl: string;
  actionsPerRequest: number;
  queueStrategy: PolicyQueueStrategy;
  prefetchThreshold?: number;
  frequency: number;
  task: string;
  robotType: string;
  stateMode: So101PolicyStateMode;
  cameraPlan: So101PolicyCameraPlan;
  resetOnTaskChange: boolean;
  clearQueueOnTaskChange: boolean;
  onPolicyTelemetry: (telemetry: PolicyTelemetry) => void;
}) {
  return (
    <>
      <SceneProbe />
      <HeldPolicyControls ctrl={heldPolicyCtrl} enabled={!policyRunning} />
      <SetupDebugBridge policyRunning={policyRunning} />
      <Debug
        showCameras={showMujocoCameraDebug}
        virtualCameras={showPolicyCameraDebug ? policyDebugCameras : []}
      />
      {policyRunning ? (
        <LeRobotPolicyRunner
          enabled={policyRunning}
          executionMode="raw-act"
          inferenceUrl={inferenceUrl}
          actionsPerRequest={actionsPerRequest}
          queueStrategy={queueStrategy}
          prefetchThreshold={prefetchThreshold}
          frequency={frequency}
          task={task}
          robotType={robotType}
          stateMode={stateMode}
          cameraPlan={cameraPlan}
          resetOnTaskChange={resetOnTaskChange}
          clearQueueOnTaskChange={clearQueueOnTaskChange}
          onTelemetry={onPolicyTelemetry}
        />
      ) : null}
    </>
  );
}

function HeldPolicyControls({
  ctrl,
  enabled,
}: {
  ctrl: number[] | null;
  enabled: boolean;
}) {
  useBeforePhysicsStep(({ data }) => {
    if (!enabled || !ctrl) return;
    for (let i = 0; i < Math.min(6, ctrl.length, data.ctrl.length); i += 1) {
      data.ctrl[i] = ctrl[i];
    }
  });

  return null;
}

function JointStateHud({
  policyRunning,
}: {
  policyRunning: boolean;
}) {
  return (
    <Card
      className="pointer-events-auto fixed bottom-4 left-4 w-[520px] max-w-[calc(100vw-32px)] rounded-lg border-slate-200/15 bg-slate-950/65 text-slate-200 shadow-2xl backdrop-blur-xl"
      size="sm"
    >
      <CardHeader className="gap-1">
        <CardTitle>SO-101 Joint State</CardTitle>
        <CardDescription>
          Policy: {policyRunning ? 'running' : 'paused'}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Collapsible>
          <CollapsibleTrigger
            render={
              <Button
                className="h-6 w-full justify-start px-0 text-xs text-slate-50"
                size="sm"
                variant="ghost"
              />
            }
          >
            <ChevronRight
              className="rotate-0"
              data-icon="inline-start"
            />
            Joint states
          </CollapsibleTrigger>
          <CollapsibleContent keepMounted>
            <div className="mt-1.5 grid gap-0.5 font-mono text-[11px]">
              {SO101_JOINT_LABELS.map((label, index) => (
                <div
                  className="grid grid-cols-[92px_82px_72px_92px] items-baseline gap-2 whitespace-nowrap"
                  key={label}
                >
                  <span>{label}</span>
                  <span data-joint-qpos={label}>q {homeJoints[index].toFixed(5)}</span>
                  <span data-joint-deg={label}>{(homeJoints[index] * 180 / Math.PI).toFixed(2)} deg</span>
                  <span data-joint-ctrl={label}>ctrl {homeJoints[index].toFixed(5)}</span>
                </div>
              ))}
            </div>
            <code
              className="mt-1.5 block max-w-full font-mono text-[11px] text-slate-200 [overflow-wrap:anywhere]"
              data-home-joints
            >
              homeJoints: [{homeJoints.map((value) => value.toFixed(6)).join(', ')}]
            </code>
          </CollapsibleContent>
        </Collapsible>
      </CardContent>
    </Card>
  );
}

function PolicyDetailSection({
  title,
  children,
  defaultOpen = false,
}: {
  title: string;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  return (
    <Collapsible className="grid gap-1" defaultOpen={defaultOpen}>
      <CollapsibleTrigger
        render={
          <Button
            className="h-6 w-full justify-start px-0 text-xs text-slate-50"
            size="sm"
            variant="ghost"
          />
        }
      >
        <ChevronRight
          className="rotate-0"
          data-icon="inline-start"
        />
        {title}
      </CollapsibleTrigger>
      <CollapsibleContent
        className="grid gap-0.5 pl-[18px] text-xs leading-snug text-slate-300 [&_code]:block [&_span]:block"
        keepMounted
      >
        {children}
      </CollapsibleContent>
    </Collapsible>
  );
}

function PolicyHud({
  policyRunning,
  onPolicyChange,
  autoPauseOnLift,
  activeTask,
  taskAfterLift,
  showPolicyCameraDebug,
  setShowPolicyCameraDebug,
  showMujocoCameraDebug,
  setShowMujocoCameraDebug,
  inferenceUrl,
  setInferenceUrl,
  actionsPerRequest,
  setActionsPerRequest,
  queueStrategy,
  prefetchThreshold,
  onToggleRun,
}: {
  policyRunning: boolean;
  onPolicyChange: (value: string) => void;
  autoPauseOnLift: boolean;
  activeTask: string;
  taskAfterLift: string;
  showPolicyCameraDebug: boolean;
  setShowPolicyCameraDebug: (value: boolean) => void;
  showMujocoCameraDebug: boolean;
  setShowMujocoCameraDebug: (value: boolean) => void;
  inferenceUrl: string;
  setInferenceUrl: (value: string) => void;
  actionsPerRequest: number;
  setActionsPerRequest: (value: number) => void;
  queueStrategy: PolicyQueueStrategy;
  prefetchThreshold?: number;
  onToggleRun: () => void;
}) {
  return (
    <Card
      className="policy-hud pointer-events-auto fixed top-4 right-4 max-h-[calc(100vh-32px)] w-[min(390px,calc(100vw-32px))] overflow-y-auto rounded-lg border-slate-200/15 bg-slate-950/65 text-slate-200 shadow-2xl backdrop-blur-xl"
      size="sm"
    >
      <CardHeader>
        <CardTitle>SO-101 Policy</CardTitle>
        <CardDescription>{policyPreset.label}</CardDescription>
        <CardAction>
          <Badge variant={policyRunning ? 'default' : 'secondary'}>
            {policyRunning ? 'Running' : 'Paused'}
          </Badge>
        </CardAction>
      </CardHeader>
      <CardContent className="grid gap-3">
        <div className="grid gap-[3px] text-xs leading-snug [&_span:first-child]:font-bold [&_span:first-child]:text-slate-50">
          <span>{policyRunning ? 'Running' : 'Paused'}</span>
          <span data-policy-source>Action source: paused</span>
          <span data-policy-camera>Cameras: waiting</span>
          <span data-policy-timing>Timing: waiting</span>
          <span data-policy-task>Task: {activeTask}</span>
          <span data-policy-remote>Remote: idle | req 0 / resp 0 | queue 0</span>
          <span data-policy-task-after-lift>After lift: {taskAfterLift || 'none'}</span>
          <span data-policy-bin-camera>Policy bin camera: {showBinInPolicyAfterLift ? 'visible after lift' : 'hidden'}</span>
          <span data-policy-autopause>Auto-pause: {autoPauseOnLift ? `on lift >= ${policyAutoPauseLiftThreshold.toFixed(3)} m` : 'off'}</span>
        </div>

        <FieldGroup className="gap-3">
          <FieldLabel className="policy-field pointer-events-auto">
            <Field className="gap-1.5">
              <FieldTitle>Policy preset</FieldTitle>
              <NativeSelect
                aria-label="Policy preset"
                className="w-full"
                disabled={policyRunning}
                onChange={(event) => onPolicyChange(event.target.value)}
                value={policyPreset.id}
              >
                <NativeSelectOption value="molmo">MolmoAct2 SO-100/101</NativeSelectOption>
                <NativeSelectOption value="act12">ACT 12D sim pick-place</NativeSelectOption>
              </NativeSelect>
            </Field>
          </FieldLabel>

          <FieldLabel className="policy-field pointer-events-auto">
            <Field className="gap-1.5">
              <FieldTitle>Inference URL</FieldTitle>
              <Input
                className="font-mono text-xs"
                value={inferenceUrl}
                onChange={(event) => setInferenceUrl(event.target.value)}
                spellCheck={false}
              />
            </Field>
          </FieldLabel>

          <FieldLabel className="policy-field pointer-events-auto">
            <Field className="gap-1.5">
              <FieldTitle>Actions per request</FieldTitle>
              <Input
                type="number"
                min={1}
                max={300}
                value={actionsPerRequest}
                onChange={(event) => {
                  const value = Number(event.target.value);
                  if (Number.isFinite(value)) {
                    setActionsPerRequest(Math.max(1, Math.min(300, Math.round(value))));
                  }
                }}
              />
            </Field>
          </FieldLabel>
        </FieldGroup>

        <Separator />

        <FieldGroup className="gap-2">
          <FieldLabel className="w-full pointer-events-auto rounded-lg border border-slate-800 bg-slate-900/55 px-3 py-2">
            <Field className="items-center" orientation="horizontal">
              <FieldContent>
                <FieldTitle>Policy camera helper</FieldTitle>
              </FieldContent>
              <Switch
                checked={showPolicyCameraDebug}
                size="sm"
                onCheckedChange={(checked) => setShowPolicyCameraDebug(checked)}
              />
            </Field>
          </FieldLabel>
          <FieldLabel className="w-full pointer-events-auto rounded-lg border border-slate-800 bg-slate-900/55 px-3 py-2">
            <Field className="items-center" orientation="horizontal">
              <FieldContent>
                <FieldTitle>MuJoCo camera debug</FieldTitle>
              </FieldContent>
              <Switch
                checked={showMujocoCameraDebug}
                size="sm"
                onCheckedChange={(checked) => setShowMujocoCameraDebug(checked)}
              />
            </Field>
          </FieldLabel>
        </FieldGroup>

        <Separator />

        <PolicyDetailSection title="Policy metadata">
          <span>Preset: {policyPreset.label}</span>
          <span>Dataset: {policyPreset.sourceRepo}</span>
          <span>Model: {policyPreset.modelId}</span>
          <span>Task: {policyTask}</span>
          <span>State: {policyPreset.stateMode}</span>
          <span data-policy-execution>Execution: raw-act</span>
          <span data-policy-model>Model: not connected</span>
          <span data-policy-space>Policy space: waiting</span>
          <span data-policy-stats>Stats: waiting</span>
          <span data-policy-observation>Observation: 0 values</span>
          <span data-policy-scheduler>Scheduler: {queueStrategy} | prefetch {prefetchThreshold ?? 'auto'}</span>
          <span data-policy-gripper>Gripper: waiting</span>
        </PolicyDetailSection>

        <PolicyDetailSection title="Scene telemetry">
          <span data-contact-count>Contacts: 0</span>
          <span data-site-position="gripperframe">gripperframe: waiting</span>
          {['red_cube', 'green_target', 'table'].map((name) => (
            <span key={name} data-object-position={name}>{name}: waiting</span>
          ))}
        </PolicyDetailSection>

        <PolicyDetailSection title="Action vectors">
          <code data-policy-action>Action: []</code>
          <code data-policy-action-units>Policy units: []</code>
          <code data-policy-model-action>Model action: []</code>
        </PolicyDetailSection>
      </CardContent>
      <CardFooter className="border-slate-200/10 bg-slate-900/35">
        <Button
          className="policy-button pointer-events-auto bg-blue-600 text-white hover:bg-blue-500 focus-visible:ring-blue-400/50"
          type="button"
          onClick={onToggleRun}
        >
          {policyRunning ? (
            <Pause data-icon="inline-start" />
          ) : (
            <Play data-icon="inline-start" />
          )}
          {policyRunning ? 'Pause policy' : 'Run policy'}
        </Button>
      </CardFooter>
    </Card>
  );
}

function So101Studio() {
  const [policyRunning, setPolicyRunning] = useState(false);
  const [heldPolicyCtrl, setHeldPolicyCtrl] = useState<number[] | null>(null);
  const [showPolicyCameraDebug, setShowPolicyCameraDebug] = useState(false);
  const [showMujocoCameraDebug, setShowMujocoCameraDebug] = useState(false);
  const [inferenceUrl, setInferenceUrl] = useState(policyPreset.inferenceUrl);
  const [actionsPerRequest, setActionsPerRequest] = useState(policyPreset.actionsPerRequest);
  const [activePolicyTask, setActivePolicyTask] = useState(policyTask);
  const startTimerRef = useRef<number | null>(null);
  const autoPauseInitialCubeZRef = useRef<number | null>(null);
  const autoPauseStableTicksRef = useRef(0);

  const onPolicyTelemetry = useCallback((telemetry: PolicyTelemetry) => {
    updatePolicyHud(telemetry);
  }, []);

  const changePolicyPreset = useCallback((value: string) => {
    const url = new URL(window.location.href);
    if (value === 'molmo') {
      url.searchParams.delete('policy');
      url.searchParams.delete('preset');
    } else {
      url.searchParams.set('policy', value);
      url.searchParams.delete('preset');
    }
    window.location.assign(url.toString());
  }, []);

  const activePolicyCameraPlan = useMemo(() => (
    policyTaskAfterLift && showBinInPolicyAfterLift && activePolicyTask === policyTaskAfterLift
      ? revealBinInPolicyCameraPlan(policyPreset.policyCamera)
      : policyPreset.policyCamera
  ), [activePolicyTask]);
  const resetPolicyOnTaskChange = booleanSearchParam(
    'resetPolicyOnTaskChange',
    !(policyPreset.id === 'molmo' && policyTaskAfterLift),
  );
  const clearQueueOnTaskChange = policyPreset.id === 'molmo' && !!policyTaskAfterLift;

  const resetSceneForPolicy = useCallback(() => {
    const debugGlobal = globalThis as typeof globalThis & CameraDebugGlobal;
    autoPauseInitialCubeZRef.current = null;
    autoPauseStableTicksRef.current = 0;
    setHeldPolicyCtrl(null);
    setActivePolicyTask(policyTask);
    debugGlobal.__so101AutoPause = {
      armed: policyAutoPauseOnLift,
      triggered: false,
      lift: 0,
      time: 0,
      qpos: [],
      ctrl: [],
      cube: null,
      gripperContacts: 0,
      movingJawContacts: 0,
    };
    debugGlobal.__so101TaskPhase = {
      initialTask: policyTask,
      activeTask: policyTask,
      taskAfterLift: policyTaskAfterLift || null,
      switched: false,
      lift: 0,
      time: 0,
    };
    setHudText('[data-policy-autopause]', `Auto-pause: ${policyAutoPauseOnLift ? 'armed' : 'off'}`);
    setHudText('[data-policy-task]', `Task: ${policyTask}`);
    setHudText('[data-policy-remote]', 'Remote: idle | req 0 / resp 0 | queue 0');
    debugGlobal.__so101SetRobotState?.(homeJoints, homeJoints);
    debugGlobal.__so101SetObjectPose?.('red_cube', redCubePosition);
  }, []);

  const toggleRun = useCallback(() => {
    if (policyRunning) {
      if (startTimerRef.current !== null) {
        window.clearTimeout(startTimerRef.current);
        startTimerRef.current = null;
      }
      setPolicyRunning(false);
      return;
    }

    if (resetSceneBeforePolicy) {
      resetSceneForPolicy();
    }
    setPolicyRunning(false);
    if (startTimerRef.current !== null) window.clearTimeout(startTimerRef.current);
    startTimerRef.current = window.setTimeout(() => {
      startTimerRef.current = null;
      setPolicyRunning(true);
    }, 200);
  }, [policyRunning, resetSceneForPolicy]);

  useEffect(() => () => {
    if (startTimerRef.current !== null) {
      window.clearTimeout(startTimerRef.current);
    }
  }, []);

  useEffect(() => {
    if (!policyRunning || (!policyAutoPauseOnLift && !policyTaskAfterLift)) return undefined;

    const interval = window.setInterval(() => {
      const debugGlobal = globalThis as typeof globalThis & CameraDebugGlobal;
      const state = debugGlobal.__so101DebugState;
      const cube = state?.bodies.red_cube;
      if (!cube) return;

      if (autoPauseInitialCubeZRef.current === null) {
        autoPauseInitialCubeZRef.current = cube[2];
      }

      const lift = cube[2] - autoPauseInitialCubeZRef.current;
      const hasCurrentGripperContact = state.contacts.some((contact) => (
        contactPairKey(contact.body1, contact.body2) === contactPairKey('red_cube', 'gripper')
      ));
      const hasCurrentMovingJawContact = state.contacts.some((contact) => (
        contactPairKey(contact.body1, contact.body2) === contactPairKey('red_cube', 'moving_jaw_so101_v1')
      ));
      const gripperContacts = state.contacts.filter((contact) => (
        contactPairKey(contact.body1, contact.body2) === contactPairKey('red_cube', 'gripper')
      )).length;
      const movingJawContacts = state.contacts.filter((contact) => (
        contactPairKey(contact.body1, contact.body2) === contactPairKey('red_cube', 'moving_jaw_so101_v1')
      )).length;
      const contactPairCounts = debugGlobal.__so101ContactPairCounts ?? {};
      const gripperContactHistory = contactPairCounts[contactPairKey('red_cube', 'gripper')] ?? 0;
      const movingJawContactHistory = contactPairCounts[contactPairKey('red_cube', 'moving_jaw_so101_v1')] ?? 0;
      const hasLiftedGraspHistory = gripperContactHistory >= 50 && movingJawContactHistory >= 20;

      const liftSwitchThreshold = policyTaskAfterLift ? policyTaskAfterLiftThreshold : policyAutoPauseLiftThreshold;
      const hasSwitchContact = policyTaskAfterLift
        ? hasLiftedGraspHistory || (hasCurrentGripperContact && hasCurrentMovingJawContact)
        : hasCurrentGripperContact && hasCurrentMovingJawContact;
      if (lift >= liftSwitchThreshold && hasSwitchContact) {
        autoPauseStableTicksRef.current += 1;
      } else {
        autoPauseStableTicksRef.current = 0;
      }

      if (autoPauseStableTicksRef.current >= policyAutoPauseStableTicks) {
        autoPauseStableTicksRef.current = 0;
        if (policyTaskAfterLift && activePolicyTask !== policyTaskAfterLift) {
          setActivePolicyTask(policyTaskAfterLift);
          debugGlobal.__so101TaskPhase = {
            initialTask: policyTask,
            activeTask: policyTaskAfterLift,
            taskAfterLift: policyTaskAfterLift,
            switched: true,
            lift,
            time: state.time,
          };
          setHudText('[data-policy-task]', `Task: ${policyTaskAfterLift}`);
          return;
        }

        if (!policyAutoPauseOnLift) return;
        setHeldPolicyCtrl(state.ctrl);
        debugGlobal.__so101AutoPause = {
          armed: true,
          triggered: true,
          lift,
          time: state.time,
          qpos: state.qpos,
          ctrl: state.ctrl,
          cube,
          gripperContacts,
          movingJawContacts,
        };
        setHudText('[data-policy-autopause]', `Auto-pause: lifted cube (${lift.toFixed(3)} m)`);
        setPolicyRunning(false);
      }
    }, 100);

    return () => window.clearInterval(interval);
  }, [activePolicyTask, policyRunning]);

  return (
    <MujocoProvider>
      <MujocoCanvas
        config={sceneConfig}
        camera={{
          position: [1.2, -1.2, 1.6],
          up: [0, 0, 1],
          fov: 45,
          near: 0.01,
          far: 100,
        }}
        dpr={[1, 2]}
        gl={{ antialias: true, powerPreference: 'high-performance' }}
        renderOptions={policyRunning ? undefined : { meshNormalSmoothing: true }}
        style={{ width: '100%', height: '100%' }}
      >
        <color attach="background" args={['#d8ddd8']} />
        <OrbitControls
          enableDamping
          dampingFactor={0.08}
          target={policyPreset.orbitTarget}
          makeDefault
        />
        <LoadingOverlay />
        <SceneChildren
          policyRunning={policyRunning}
          heldPolicyCtrl={heldPolicyCtrl}
          showPolicyCameraDebug={showPolicyCameraDebug}
          showMujocoCameraDebug={showMujocoCameraDebug}
          inferenceUrl={inferenceUrl}
          actionsPerRequest={actionsPerRequest}
          queueStrategy={policyQueueStrategy}
          prefetchThreshold={policyPrefetchThreshold}
          frequency={policyPreset.frequency}
          task={activePolicyTask}
          robotType={policyPreset.robotType}
          stateMode={policyPreset.stateMode}
          cameraPlan={activePolicyCameraPlan}
          resetOnTaskChange={resetPolicyOnTaskChange}
          clearQueueOnTaskChange={clearQueueOnTaskChange}
          onPolicyTelemetry={onPolicyTelemetry}
        />
        <ScenarioLighting preset="studio" intensity={1.55} />
      </MujocoCanvas>
      <JointStateHud policyRunning={policyRunning} />
      <PolicyHud
        policyRunning={policyRunning}
        onPolicyChange={changePolicyPreset}
        autoPauseOnLift={policyAutoPauseOnLift}
        activeTask={activePolicyTask}
        taskAfterLift={policyTaskAfterLift}
        showPolicyCameraDebug={showPolicyCameraDebug}
        setShowPolicyCameraDebug={setShowPolicyCameraDebug}
        showMujocoCameraDebug={showMujocoCameraDebug}
        setShowMujocoCameraDebug={setShowMujocoCameraDebug}
        inferenceUrl={inferenceUrl}
        setInferenceUrl={setInferenceUrl}
        actionsPerRequest={actionsPerRequest}
        setActionsPerRequest={setActionsPerRequest}
        queueStrategy={policyQueueStrategy}
        prefetchThreshold={policyPrefetchThreshold}
        onToggleRun={toggleRun}
      />
    </MujocoProvider>
  );
}

const rootRoute = createRootRoute({
  component: () => <Outlet />,
});

const studioRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  component: So101Studio,
});

const routeTree = rootRoute.addChildren([studioRoute]);

const router = createRouter({ routeTree });

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router;
  }
}

export function App() {
  return <RouterProvider router={router} />;
}
