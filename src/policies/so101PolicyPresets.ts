import type { CreatePolicyCameraFrameCapturePlanOptions, SceneObject, XmlPatch } from 'mujoco-react';
import type { PolicyQueueStrategy } from '../controllers/LeRobotPickPlacePolicy';

export type So101PolicyStateMode = 'cube-to-target-12';
export type So101PolicyPresetId = 'act12' | 'molmo';
export type So101PolicyCameraPlan = Omit<
  CreatePolicyCameraFrameCapturePlanOptions,
  'cameras' | 'sites' | 'bodies'
>;

export interface So101PolicyPreset {
  id: So101PolicyPresetId;
  label: string;
  sceneFile: string;
  modelId: string;
  sourceRepo: string;
  inferenceUrl: string;
  task: string;
  robotType: string;
  homeJoints: number[];
  stateMode: So101PolicyStateMode;
  actionsPerRequest: number;
  queueStrategy: PolicyQueueStrategy;
  prefetchThreshold?: number;
  frequency: number;
  policyCamera: So101PolicyCameraPlan;
  orbitTarget: [number, number, number];
  xmlPatches: XmlPatch[];
  redCubePosition: [number, number, number];
  redCubeSize: [number, number, number];
  tablePosition: [number, number, number];
  tableSize: [number, number, number];
  greenTargetPosition: [number, number, number];
  redCubeRgba: [number, number, number, number];
  redCubeMass?: number;
  redCubeFriction?: string;
  redCubeCondim?: number;
  greenTargetSize: [number, number, number];
  greenTargetRgba: [number, number, number, number];
}

export interface So101SceneObjectOptions {
  redCubeMass?: number;
  redCubeFriction?: string;
  redCubeSolref?: string;
  redCubeSolimp?: string;
  redCubeCondim?: number;
  includeAct12BinWalls?: boolean;
}

export const ACT_12D_HOME_JOINTS = [
  -0.117329,
  -1.727294,
  1.69,
  -1.629534,
  -0.058387,
  -0.17453,
];

export const MOLMO_SO101_HOME_JOINTS = [
  0.069314,
  -1.685636,
  0.810382,
  1.600638,
  -1.396152,
  -0.17453,
];

const ACT_12D_XML_PATCHES: XmlPatch[] = [
  {
    target: 'SO101.xml',
    replace: [
      '<body name="base" pos="0 0 0" quat="1 0 0 0" childclass="so101">',
      '<body name="base" pos="0.18 0 0.42" quat="1 0 0 0" childclass="so101">',
    ],
  },
  {
    target: 'SO101.xml',
    replace: [
      '<geom group="3" condim="3"/>',
      '<geom group="3" condim="3" contype="0" conaffinity="0"/>',
    ],
  },
  {
    target: 'SO101.xml',
    replace: [
      '<geom name="fixed_jaw_box6" class="collision_gripper" type="box" size="0.001 0.009 0.008"',
      '<geom name="fixed_jaw_box6" class="collision_gripper" type="box" size="0.006 0.014 0.012"',
    ],
  },
  {
    target: 'SO101.xml',
    replace: [
      '<geom name="fixed_jaw_box7" class="collision_gripper" type="box" size="0.001 0.01 0.008"',
      '<geom name="fixed_jaw_box7" class="collision_gripper" type="box" size="0.006 0.015 0.012"',
    ],
  },
  {
    target: 'SO101.xml',
    replace: [
      '<geom name="moving_jaw_box2" class="collision_gripper" type="box" size="0.001 0.004 0.004"',
      '<geom name="moving_jaw_box2" class="collision_gripper" type="box" size="0.008 0.014 0.012"',
    ],
  },
  {
    target: 'SO101.xml',
    replace: [
      '<geom name="moving_jaw_box3" class="collision_gripper" type="box" size="0.001 0.005 0.006"',
      '<geom name="moving_jaw_box3" class="collision_gripper" type="box" size="0.008 0.014 0.012"',
    ],
  },
  {
    target: 'SO101.xml',
    replace: [
      '<geom name="camera_box1" class="collision_gripper" type="box" size="0.015 0.015 0.003"',
      '<geom name="camera_box1" class="collision_gripper" contype="0" conaffinity="0" type="box" size="0.015 0.015 0.003"',
    ],
  },
  {
    target: 'SO101.xml',
    replace: [
      '<geom name="camera_box2" class="collision_gripper" type="box" size="0.021 0.021 0.003"',
      '<geom name="camera_box2" class="collision_gripper" contype="0" conaffinity="0" type="box" size="0.021 0.021 0.003"',
    ],
  },
];

const CAPTURE_HIDDEN_GEOM_GROUPS = [3, 4] as const;
const CAPTURE_HIDDEN_FLOOR_GEOMS = ['floor', 'floor_box_geom'] as const;

export const ACT12_POLICY_CAMERA = {
  position: [0.72, 0, 1.08],
  lookAt: [0.4, 0, 0.43],
  up: [0, 0, 1],
  fov: 50,
  width: 640,
  height: 480,
} as const;

const ACT12_POLICY_CAMERA_PLAN: So101PolicyCameraPlan = {
  cameraKeys: ['front'],
  requireAll: true,
  defaults: {
    width: 640,
    type: 'image/jpeg',
    quality: 0.82,
    hiddenGeomGroups: CAPTURE_HIDDEN_GEOM_GROUPS,
    hiddenGeomNames: CAPTURE_HIDDEN_FLOOR_GEOMS,
    background: '#6c6c6c',
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
    height: ACT12_POLICY_CAMERA.height,
    fov: ACT12_POLICY_CAMERA.fov,
  },
  streamOptions: {
    front: {
      position: [...ACT12_POLICY_CAMERA.position],
      lookAt: [...ACT12_POLICY_CAMERA.lookAt],
      up: [...ACT12_POLICY_CAMERA.up],
      fov: ACT12_POLICY_CAMERA.fov,
    },
  },
};

export const MOLMO_TOP_POLICY_CAMERA = {
  position: [0.64, -0.46, 0.62],
  lookAt: [0.33, 0.04, 0.48],
  up: [0, 0, 1],
  fov: 48,
  width: 640,
  height: 480,
} as const;

export const MOLMO_SIDE_POLICY_CAMERA = {
  position: [0.38, -0.12, 0.86],
  lookAt: [0.32, 0.04, 0.44],
  up: [0, 0, 1],
  fov: 42,
  width: 640,
  height: 480,
} as const;

const MOLMO_POLICY_CAMERA_PLAN: So101PolicyCameraPlan = {
  cameraKeys: ['top', 'side'],
  requireAll: true,
  includeObservationImageAliases: false,
  defaults: {
    width: 640,
    height: 480,
    type: 'image/jpeg',
    quality: 0.86,
    hiddenGeomGroups: CAPTURE_HIDDEN_GEOM_GROUPS,
    hiddenGeomNames: CAPTURE_HIDDEN_FLOOR_GEOMS,
    background: '#d8ddd8',
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
  },
  streamOptions: {
    top: {
      position: [...MOLMO_TOP_POLICY_CAMERA.position],
      lookAt: [...MOLMO_TOP_POLICY_CAMERA.lookAt],
      up: [...MOLMO_TOP_POLICY_CAMERA.up],
      fov: MOLMO_TOP_POLICY_CAMERA.fov,
    },
    side: {
      position: [...MOLMO_SIDE_POLICY_CAMERA.position],
      lookAt: [...MOLMO_SIDE_POLICY_CAMERA.lookAt],
      up: [...MOLMO_SIDE_POLICY_CAMERA.up],
      fov: MOLMO_SIDE_POLICY_CAMERA.fov,
    },
  },
};

export const SO101_POLICY_PRESET: So101PolicyPreset = {
  id: 'act12',
  label: 'ACT 12D sim pick-place',
  sceneFile: 'SO101.xml',
  modelId: 'davidlinjiahao/lerobot_so101_base_sim_pickplace',
  sourceRepo: 'davidlinjiahao/lerobot_batch_001',
  inferenceUrl: 'http://127.0.0.1:8776/infer',
  task: 'pick and place the cube',
  robotType: 'so101',
  homeJoints: ACT_12D_HOME_JOINTS,
  stateMode: 'cube-to-target-12',
  actionsPerRequest: 100,
  queueStrategy: 'append',
  frequency: 30,
  policyCamera: ACT12_POLICY_CAMERA_PLAN,
  orbitTarget: [0.4, 0.02, 0.42],
  xmlPatches: ACT_12D_XML_PATCHES,
  redCubePosition: [0.4, 0.18, 0.432493],
  redCubeSize: [0.0125, 0.0125, 0.0125],
  tablePosition: [0.48, 0, 0.02],
  tableSize: [0.22, 0.42, 0.4],
  greenTargetPosition: [0.4, -0.18, 0.42],
  redCubeRgba: [0.92, 0.92, 0.88, 1],
  greenTargetSize: [0.09, 0.09, 0.004],
  greenTargetRgba: [0.45, 0.28, 0.05, 0.45],
};

export const MOLMO_SO101_POLICY_PRESET: So101PolicyPreset = {
  id: 'molmo',
  label: 'MolmoAct2 SO-100/101',
  sceneFile: 'SO101.xml',
  modelId: 'allenai/MolmoAct2-SO100_101',
  sourceRepo: 'allenai/MolmoAct2-SO100_101',
  inferenceUrl: 'https://vwz196x8czmhbn-8000.proxy.runpod.net/infer',
  task: 'pick up the red cube',
  robotType: 'so101',
  homeJoints: MOLMO_SO101_HOME_JOINTS,
  stateMode: 'cube-to-target-12',
  actionsPerRequest: 45,
  queueStrategy: 'replace',
  prefetchThreshold: 1,
  frequency: 30,
  policyCamera: MOLMO_POLICY_CAMERA_PLAN,
  orbitTarget: [0.32, 0.02, 0.45],
  xmlPatches: ACT_12D_XML_PATCHES,
  redCubePosition: [0.36, 0.06, 0.44],
  redCubeSize: [0.02, 0.02, 0.02],
  tablePosition: [0.4, 0, 0.02],
  tableSize: [0.34, 0.42, 0.4],
  greenTargetPosition: [0.4, -0.18, 0.42],
  redCubeRgba: [0.9, 0.12, 0.08, 1],
  redCubeMass: 0.006,
  redCubeFriction: '4 0.8 0.08',
  redCubeCondim: 6,
  greenTargetSize: [0.09, 0.09, 0.004],
  greenTargetRgba: [0.45, 0.28, 0.05, 0.45],
};

export function selectSo101PolicyPreset(value: string | null | undefined): So101PolicyPreset {
  return value === 'act12' ? SO101_POLICY_PRESET : MOLMO_SO101_POLICY_PRESET;
}

export function createSo101SceneObjects(
  preset: So101PolicyPreset,
  redCubePosition: [number, number, number],
  redCubeSize: [number, number, number],
  options: So101SceneObjectOptions = {},
): SceneObject[] {
  const objects: SceneObject[] = [
    {
      name: 'table',
      type: 'box',
      size: preset.tableSize,
      position: preset.tablePosition,
      rgba: [0.78, 0.46, 0.22, 1],
      friction: '1.5 0.3 0.1',
      solref: '0.01 1',
      solimp: '0.95 0.99 0.001 0.5 2',
      condim: 4,
    },
    {
      name: 'red_cube',
      type: 'box',
      size: redCubeSize,
      position: redCubePosition,
      rgba: preset.redCubeRgba,
      mass: options.redCubeMass ?? preset.redCubeMass ?? 0.01,
      freejoint: true,
      friction: options.redCubeFriction ?? preset.redCubeFriction ?? '1 0.05 0.001',
      solref: options.redCubeSolref ?? '0.01 1',
      solimp: options.redCubeSolimp ?? '0.95 0.99 0.001 0.5 2',
      condim: options.redCubeCondim ?? preset.redCubeCondim ?? 4,
    },
    {
      name: 'green_target',
      type: 'box',
      size: preset.greenTargetSize,
      position: preset.greenTargetPosition,
      rgba: preset.greenTargetRgba,
      friction: '2 0.3 0.1',
      solref: '0.01 1',
      solimp: '0.95 0.99 0.001 0.5 2',
      condim: 4,
    },
  ];

  if (options.includeAct12BinWalls === false) return objects;

  return [
    ...objects,
    {
      name: 'target_bin_left_wall',
      type: 'box',
      size: [0.006, 0.09, 0.045],
      position: [
        preset.greenTargetPosition[0] - 0.09,
        preset.greenTargetPosition[1],
        preset.greenTargetPosition[2] + 0.045,
      ],
      rgba: [preset.greenTargetRgba[0], preset.greenTargetRgba[1], preset.greenTargetRgba[2], 0.5],
      friction: '2 0.3 0.1',
      solref: '0.01 1',
      solimp: '0.95 0.99 0.001 0.5 2',
      condim: 4,
    },
    {
      name: 'target_bin_right_wall',
      type: 'box',
      size: [0.006, 0.09, 0.045],
      position: [
        preset.greenTargetPosition[0] + 0.09,
        preset.greenTargetPosition[1],
        preset.greenTargetPosition[2] + 0.045,
      ],
      rgba: [preset.greenTargetRgba[0], preset.greenTargetRgba[1], preset.greenTargetRgba[2], 0.5],
      friction: '2 0.3 0.1',
      solref: '0.01 1',
      solimp: '0.95 0.99 0.001 0.5 2',
      condim: 4,
    },
    {
      name: 'target_bin_front_wall',
      type: 'box',
      size: [0.09, 0.006, 0.045],
      position: [
        preset.greenTargetPosition[0],
        preset.greenTargetPosition[1] - 0.09,
        preset.greenTargetPosition[2] + 0.045,
      ],
      rgba: [preset.greenTargetRgba[0], preset.greenTargetRgba[1], preset.greenTargetRgba[2], 0.5],
      friction: '2 0.3 0.1',
      solref: '0.01 1',
      solimp: '0.95 0.99 0.001 0.5 2',
      condim: 4,
    },
    {
      name: 'target_bin_back_wall',
      type: 'box',
      size: [0.09, 0.006, 0.045],
      position: [
        preset.greenTargetPosition[0],
        preset.greenTargetPosition[1] + 0.09,
        preset.greenTargetPosition[2] + 0.045,
      ],
      rgba: [preset.greenTargetRgba[0], preset.greenTargetRgba[1], preset.greenTargetRgba[2], 0.5],
      friction: '2 0.3 0.1',
      solref: '0.01 1',
      solimp: '0.95 0.99 0.001 0.5 2',
      condim: 4,
    },
  ];
}
