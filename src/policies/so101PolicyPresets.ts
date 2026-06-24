import type { SceneObject, XmlPatch } from 'mujoco-react';
import type { PolicyQueueStrategy } from '../controllers/LeRobotPickPlacePolicy';

export type So101PolicyStateMode = 'cube-to-target-12';

export interface So101PolicyPreset {
  id: 'act12';
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
  orbitTarget: [number, number, number];
  xmlPatches: XmlPatch[];
  redCubePosition: [number, number, number];
  redCubeSize: [number, number, number];
  tablePosition: [number, number, number];
  tableSize: [number, number, number];
  greenTargetPosition: [number, number, number];
  redCubeRgba: [number, number, number, number];
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
];

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
      mass: options.redCubeMass ?? 0.01,
      freejoint: true,
      friction: options.redCubeFriction ?? '1 0.05 0.001',
      solref: options.redCubeSolref ?? '0.01 1',
      solimp: options.redCubeSolimp ?? '0.95 0.99 0.001 0.5 2',
      condim: options.redCubeCondim ?? 4,
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
