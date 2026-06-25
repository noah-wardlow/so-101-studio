#!/usr/bin/env node
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { chromium } from 'playwright';

function readArg(name, fallback) {
  const prefix = `--${name}=`;
  const match = process.argv.find((arg) => arg.startsWith(prefix));
  if (!match) return process.env[name.toUpperCase().replaceAll('-', '_')] ?? fallback;
  return match.slice(prefix.length);
}

function readNumberArg(name, fallback) {
  const value = Number(readArg(name, String(fallback)));
  return Number.isFinite(value) ? value : fallback;
}

function assert(condition, message, failures) {
  if (!condition) failures.push(message);
}

const url = readArg('url', 'http://127.0.0.1:3001/');
const durationMs = readNumberArg('duration-ms', 20000);
const minResponses = readNumberArg('min-responses', 3);
const maxErrors = readNumberArg('max-errors', 0);
const maxTargetDistance = readNumberArg('max-target-distance', 0.09);
const minCubeLift = readNumberArg('min-cube-lift', 0);
const minTargetContactCount = readNumberArg('min-target-contact-count', 0);
const minGripperContactCount = readNumberArg('min-gripper-contact-count', 0);
const minMovingJawContactCount = readNumberArg('min-moving-jaw-contact-count', 0);
const minFinalCubeLift = readNumberArg('min-final-cube-lift', -Infinity);
const sampleIntervalMs = readNumberArg('sample-interval-ms', 250);
const objectBody = readArg('object-body', 'red_cube');
const targetBody = readArg('target-body', 'green_target');
const stackSpecText = readArg('stack', '');
const maxStackXYDistance = readNumberArg('max-stack-xy-distance', Infinity);
const minStackZDelta = readNumberArg('min-stack-z-delta', -Infinity);
const inferenceUrl = readArg('inference-url', '');
const screenshotPath = readArg('screenshot', '');
const reportPath = readArg('report', '');
const cameraFrameDir = readArg('camera-frame-dir', '');
const initialBodyPoseText = readArg('initial-body-pose', '');
const initialRobotStateText = readArg('initial-robot-state', '');
const actionsPerRequestOverride = readArg('actions-per-request', '');
const frequencyOverride = readArg('frequency', '');
const requireCameras = readArg('require-cameras', 'true') !== 'false';

function parseStackSpec(value) {
  if (!value) return null;
  const match = value.match(/^([^:]+):on:([^:]+)$/);
  if (!match) throw new Error(`Invalid --stack value "${value}". Use top:on:base, for example box2:on:box.`);
  return { top: match[1], base: match[2] };
}

function parseInitialBodyPose(value) {
  if (!value) return null;
  const match = value.match(/^([^:]+):([^,]+),([^,]+),([^,]+)(?::([^,]+),([^,]+),([^,]+),([^,]+))?$/);
  if (!match) {
    throw new Error(`Invalid --initial-body-pose value "${value}". Use body:x,y,z or body:x,y,z:w,x,y,z.`);
  }
  const position = [Number(match[2]), Number(match[3]), Number(match[4])];
  const quaternion = match[5] === undefined
    ? undefined
    : [Number(match[5]), Number(match[6]), Number(match[7]), Number(match[8])];
  if (!position.every(Number.isFinite) || (quaternion && !quaternion.every(Number.isFinite))) {
    throw new Error(`Invalid numeric value in --initial-body-pose "${value}".`);
  }
  return { bodyName: match[1], position, quaternion };
}

function parseInitialRobotState(value) {
  if (!value) return null;
  const qpos = value.split(',').map((part) => Number(part.trim()));
  if (qpos.length < 6 || !qpos.every(Number.isFinite)) {
    throw new Error(`Invalid --initial-robot-state value "${value}". Use six comma-separated qpos values.`);
  }
  return qpos;
}

function distance3(a, b) {
  if (!a || !b) return null;
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

function distanceXY(a, b) {
  if (!a || !b) return null;
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function contactPairKey(bodyA, bodyB) {
  return [bodyA, bodyB].sort().join('::');
}

function contactPairCount(contactPairCounts, bodyA, bodyB) {
  return contactPairCounts?.[contactPairKey(bodyA, bodyB)] ?? 0;
}

function dataUrlToBytes(value) {
  const match = /^data:([^;,]+)?(?:;[^,]+)?,(.*)$/s.exec(value);
  if (!match) return null;
  return {
    mimeType: match[1] || 'application/octet-stream',
    bytes: Buffer.from(match[2], 'base64'),
  };
}

function extensionForMimeType(mimeType) {
  if (mimeType === 'image/png') return 'png';
  if (mimeType === 'image/webp') return 'webp';
  return 'jpg';
}

function summarizeTrajectory(samples, initialBodies, finalBodies, objectName) {
  const initialObject = initialBodies?.[objectName] ?? null;
  const objectSamples = samples
    .map((sample) => ({ time: sample.time, position: sample.bodies?.[objectName] ?? null }))
    .filter((sample) => sample.position);
  const gripperSamples = samples
    .map((sample) => ({ time: sample.time, position: sample.bodies?.gripper ?? null }))
    .filter((sample) => sample.position);
  const movingJawSamples = samples
    .map((sample) => ({ time: sample.time, position: sample.bodies?.moving_jaw_so101_v1 ?? null }))
    .filter((sample) => sample.position);
  const closestGripperObject = samples.reduce((best, sample) => {
    const object = sample.bodies?.[objectName] ?? null;
    const gripper = sample.bodies?.gripper ?? null;
    const distance = distance3(object, gripper);
    if (distance === null) return best;
    if (!best || distance < best.distance) {
      return { time: sample.time, distance, object, gripper };
    }
    return best;
  }, null);
  const closestMovingJawObject = samples.reduce((best, sample) => {
    const object = sample.bodies?.[objectName] ?? null;
    const movingJaw = sample.bodies?.moving_jaw_so101_v1 ?? null;
    const distance = distance3(object, movingJaw);
    if (distance === null) return best;
    if (!best || distance < best.distance) {
      return { time: sample.time, distance, object, movingJaw };
    }
    return best;
  }, null);
  const maxObjectZSample = objectSamples.reduce((best, sample) => (
    !best || sample.position[2] > best.position[2] ? sample : best
  ), null);
  const finalObject = finalBodies?.[objectName] ?? null;

  return {
    sampleCount: samples.length,
    objectBody: objectName,
    object: {
      initial: initialObject,
      final: finalObject,
      maxZ: maxObjectZSample?.position[2] ?? null,
      maxZAt: maxObjectZSample?.time ?? null,
      liftFromInitial: initialObject && maxObjectZSample
        ? maxObjectZSample.position[2] - initialObject[2]
        : null,
      finalLiftFromInitial: initialObject && finalObject
        ? finalObject[2] - initialObject[2]
        : null,
    },
    gripper: {
      maxZ: gripperSamples.reduce((max, sample) => Math.max(max, sample.position[2]), -Infinity),
      minZ: gripperSamples.reduce((min, sample) => Math.min(min, sample.position[2]), Infinity),
    },
    movingJaw: {
      maxZ: movingJawSamples.reduce((max, sample) => Math.max(max, sample.position[2]), -Infinity),
      minZ: movingJawSamples.reduce((min, sample) => Math.min(min, sample.position[2]), Infinity),
    },
    closestGripperObject,
    closestMovingJawObject,
  };
}

function summarizeBodyMotion(initialBodies, finalBodies) {
  const names = new Set([
    ...Object.keys(initialBodies ?? {}),
    ...Object.keys(finalBodies ?? {}),
  ]);
  return Object.fromEntries(Array.from(names).sort().map((name) => [
    name,
    {
      initial: initialBodies?.[name] ?? null,
      final: finalBodies?.[name] ?? null,
      distance: distance3(initialBodies?.[name], finalBodies?.[name]),
    },
  ]));
}

function summarizeStack(finalBodies, contactPairCounts, stackSpec) {
  if (!stackSpec) return null;
  const top = finalBodies?.[stackSpec.top] ?? null;
  const base = finalBodies?.[stackSpec.base] ?? null;
  return {
    topBody: stackSpec.top,
    baseBody: stackSpec.base,
    top,
    base,
    xyDistance: distanceXY(top, base),
    zDelta: top && base ? top[2] - base[2] : null,
    contactCount: contactPairCounts?.[contactPairKey(stackSpec.top, stackSpec.base)] ?? 0,
  };
}

const stackSpec = parseStackSpec(stackSpecText);
const initialBodyPose = parseInitialBodyPose(initialBodyPoseText);
const initialRobotState = parseInitialRobotState(initialRobotStateText);

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
const pageErrors = [];
const consoleErrors = [];

page.on('pageerror', (error) => {
  pageErrors.push(error.stack || error.message);
});
page.on('console', (message) => {
  if (message.type() === 'error') consoleErrors.push(message.text());
});

try {
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForSelector('button.policy-button', { timeout: 60000 });
  await page.waitForFunction(() => globalThis.__so101DebugState?.bodies, { timeout: 60000 });

  if (initialRobotState) {
    await page.evaluate(async (qpos) => {
      if (typeof globalThis.__so101SetRobotState !== 'function') {
        throw new Error('__so101SetRobotState is not available.');
      }
      globalThis.__so101SetRobotState(qpos, qpos);
      await new Promise((resolve) => setTimeout(resolve, 250));
    }, initialRobotState);
  }

  if (initialBodyPose) {
    await page.evaluate(async (pose) => {
      if (typeof globalThis.__so101SetObjectPose !== 'function') {
        throw new Error('__so101SetObjectPose is not available.');
      }
      globalThis.__so101SetObjectPose(pose.bodyName, pose.position, pose.quaternion);
      await new Promise((resolve) => setTimeout(resolve, 250));
    }, initialBodyPose);
  }

  const before = await page.evaluate(() => ({
    buttonText: document.querySelector('button.policy-button')?.textContent ?? null,
    smoothedVisible: document.body.textContent?.includes('Smoothed policy controls') ?? false,
    preset: Array.from(document.querySelectorAll('.policy-hud span'))
      .map((element) => element.textContent)
      .find((text) => text?.startsWith('Preset:')) ?? null,
  }));

  const initialState = await page.evaluate(() => {
    const state = globalThis.__so101DebugState ?? null;
    return state ? {
      time: state.time,
      qpos: state.qpos,
      ctrl: state.ctrl,
      bodies: state.bodies,
      contacts: state.contacts,
      ncon: state.ncon,
    } : null;
  });

  if (before.buttonText?.includes('Run')) {
    if (inferenceUrl) {
      await page.fill('label.policy-field input', inferenceUrl);
    }
    if (actionsPerRequestOverride) {
      const actionsInput = page
        .locator('label.policy-field')
        .filter({ hasText: 'Actions per request' })
        .locator('input');
      await actionsInput.fill(actionsPerRequestOverride);
    }
    if (frequencyOverride) {
      const frequencyInput = page
        .locator('label.policy-field')
        .filter({ hasText: 'Frequency' })
        .locator('input');
      if (await frequencyInput.count()) {
        await frequencyInput.fill(frequencyOverride);
      }
    }
    await page.click('button.policy-button');
  }

  const trajectorySamples = [];
  const sampleStartedAt = Date.now();
  while (Date.now() - sampleStartedAt < durationMs) {
    await page.waitForTimeout(Math.max(50, sampleIntervalMs));
    trajectorySamples.push(await page.evaluate(() => {
      const state = globalThis.__so101DebugState ?? null;
      return state ? {
        wallTime: Date.now(),
        time: state.time,
        qpos: state.qpos,
        ctrl: state.ctrl,
        bodies: state.bodies,
        contacts: state.contacts,
        ncon: state.ncon,
      } : null;
    }));
  }

  const result = await page.evaluate(({ objectBody, targetBody }) => {
    const policy = globalThis.__lerobotPolicyDebug ?? [];
    const state = globalThis.__so101DebugState ?? null;
    const cameraFrames = globalThis.__lerobotCameraFrames ?? null;
    const autoPause = globalThis.__so101AutoPause ?? null;
    const taskPhase = globalThis.__so101TaskPhase ?? null;
    const contactHistory = globalThis.__so101ContactHistory ?? [];
    const contactPairCounts = { ...(globalThis.__so101ContactPairCounts ?? {}) };
    for (const contact of contactHistory) {
      const key = [contact.body1, contact.body2].sort().join('::');
      if (contactPairCounts[key] === undefined) {
        contactPairCounts[key] = 0;
      }
    }
    const object = state?.bodies?.[objectBody] ?? null;
    const target = state?.bodies?.[targetBody] ?? null;
    const objectTargetDistance = object && target
      ? Math.hypot(object[0] - target[0], object[1] - target[1], object[2] - target[2])
      : null;
    const requests = policy.filter((entry) => entry.kind === 'request');
    const responses = policy.filter((entry) => entry.kind === 'response');
    const errors = policy.filter((entry) => entry.kind === 'error');

    return {
      requests: requests.length,
      responses: responses.length,
      errors: errors.map((entry) => entry.error),
      autoPause,
      taskPhase,
      lastRequest: requests.at(-1) ?? null,
      lastResponse: responses.at(-1) ?? null,
      cameraSource: document.querySelector('[data-policy-camera]')?.textContent ?? null,
      actionSource: document.querySelector('[data-policy-source]')?.textContent ?? null,
      execution: document.querySelector('[data-policy-execution]')?.textContent ?? null,
      cameraKeys: cameraFrames ? Object.keys(cameraFrames).filter((key) => key !== 'at') : [],
      cameraDataLengths: cameraFrames
        ? Object.fromEntries(
          Object.entries(cameraFrames)
            .filter(([key]) => key !== 'at')
            .map(([key, value]) => [key, typeof value === 'string' ? value.length : 0])
        )
        : {},
      objectBody,
      targetBody,
      object,
      red: objectBody === 'red_cube' ? object : state?.bodies?.red_cube ?? null,
      target,
      objectTargetDistance,
      redTargetDistance: objectBody === 'red_cube' && targetBody === 'green_target'
        ? objectTargetDistance
        : null,
      bodies: state?.bodies ?? {},
      currentContacts: state?.contacts?.slice(0, 12) ?? [],
      contactHistoryTail: contactHistory.slice(-12).map((contact) => ({
        body1: contact.body1,
        body2: contact.body2,
        geom1: contact.geom1,
        geom2: contact.geom2,
        time: contact.time,
        dist: contact.dist,
      })),
      contactPairCounts,
      contactHistoryCount: contactHistory.length,
      objectTargetContactCount: contactPairCounts[[objectBody, targetBody].sort().join('::')] ?? 0,
      redTargetContactCount: contactPairCounts[['red_cube', 'green_target'].sort().join('::')] ?? 0,
    };
  }, { objectBody, targetBody });

  if (screenshotPath) {
    await page.screenshot({ path: screenshotPath, fullPage: true });
  }

  const exportedCameraFrames = cameraFrameDir
    ? await page.evaluate(() => ({
      latest: globalThis.__lerobotCameraFrames ?? null,
      history: globalThis.__lerobotCameraFrameHistory ?? [],
    }))
    : null;

  const bodyMotion = summarizeBodyMotion(initialState?.bodies, result.bodies);
  const trajectory = summarizeTrajectory(
    trajectorySamples.filter(Boolean),
    initialState?.bodies,
    result.bodies,
    objectBody,
  );
  const stack = summarizeStack(result.bodies, result.contactPairCounts, stackSpec);
  const failures = [];
  assert(!before.smoothedVisible, 'Hybrid/smoothed policy option is visible.', failures);
  assert(pageErrors.length === 0, `Page errors: ${pageErrors.join(' | ')}`, failures);
  assert(consoleErrors.length === 0, `Console errors: ${consoleErrors.join(' | ')}`, failures);
  assert(result.responses >= minResponses, `Expected at least ${minResponses} policy responses, got ${result.responses}.`, failures);
  assert(result.errors.length <= maxErrors, `Expected at most ${maxErrors} policy errors, got ${result.errors.length}: ${result.errors.join(' | ')}`, failures);
  if (requireCameras) {
    assert(result.cameraKeys.length > 0, 'No policy camera frames captured.', failures);
    assert(
      Object.values(result.cameraDataLengths).every((length) => length > 1000),
      `Camera frame payloads look empty: ${JSON.stringify(result.cameraDataLengths)}`,
      failures
    );
  }
  if (result.objectTargetDistance !== null) {
    assert(
      result.objectTargetDistance <= maxTargetDistance,
      `${objectBody} to ${targetBody} distance ${result.objectTargetDistance.toFixed(4)} exceeds ${maxTargetDistance}.`,
      failures
    );
  }
  if (minCubeLift > 0) {
    assert(
      trajectory.object.liftFromInitial !== null && trajectory.object.liftFromInitial >= minCubeLift,
      `${objectBody} lift ${trajectory.object.liftFromInitial?.toFixed(4) ?? 'missing'} is below ${minCubeLift}.`,
      failures
    );
  }
  if (Number.isFinite(minFinalCubeLift)) {
    assert(
      trajectory.object.finalLiftFromInitial !== null && trajectory.object.finalLiftFromInitial >= minFinalCubeLift,
      `${objectBody} final lift ${trajectory.object.finalLiftFromInitial?.toFixed(4) ?? 'missing'} is below ${minFinalCubeLift}.`,
      failures
    );
  }
  if (minTargetContactCount > 0) {
    assert(
      result.objectTargetContactCount >= minTargetContactCount,
      `${objectBody} to ${targetBody} contact count ${result.objectTargetContactCount} is below ${minTargetContactCount}.`,
      failures
    );
  }
  const gripperContactCount = contactPairCount(result.contactPairCounts, objectBody, 'gripper');
  const movingJawContactCount = contactPairCount(result.contactPairCounts, objectBody, 'moving_jaw_so101_v1');
  if (minGripperContactCount > 0) {
    assert(
      gripperContactCount >= minGripperContactCount,
      `${objectBody} to gripper contact count ${gripperContactCount} is below ${minGripperContactCount}.`,
      failures
    );
  }
  if (minMovingJawContactCount > 0) {
    assert(
      movingJawContactCount >= minMovingJawContactCount,
      `${objectBody} to moving_jaw_so101_v1 contact count ${movingJawContactCount} is below ${minMovingJawContactCount}.`,
      failures
    );
  }
  if (stackSpec) {
    assert(stack?.xyDistance !== null, `Missing bodies for stack check ${stackSpec.top}:on:${stackSpec.base}.`, failures);
    if (stack?.xyDistance !== null && Number.isFinite(maxStackXYDistance)) {
      assert(
        stack.xyDistance <= maxStackXYDistance,
        `Stack XY distance ${stack.xyDistance.toFixed(4)} exceeds ${maxStackXYDistance}.`,
        failures
      );
    }
    if (stack?.zDelta !== null && Number.isFinite(minStackZDelta)) {
      assert(
        stack.zDelta >= minStackZDelta,
        `Stack Z delta ${stack.zDelta.toFixed(4)} is below ${minStackZDelta}.`,
        failures
      );
    }
  }

  const report = {
    ok: failures.length === 0,
    url,
    durationMs,
    thresholds: {
      minResponses,
      maxErrors,
      maxTargetDistance,
      minCubeLift,
      minTargetContactCount,
      minGripperContactCount,
      minMovingJawContactCount,
      minFinalCubeLift: Number.isFinite(minFinalCubeLift) ? minFinalCubeLift : null,
      sampleIntervalMs,
      objectBody,
      targetBody,
      initialBodyPose,
      initialRobotState,
      actionsPerRequestOverride: actionsPerRequestOverride || null,
      frequencyOverride: frequencyOverride || null,
      requireCameras,
      stack: stackSpec,
      maxStackXYDistance,
      minStackZDelta,
    },
    inferenceUrl: inferenceUrl || null,
    before,
    initialState,
    result,
    gripperContactCount,
    movingJawContactCount,
    bodyMotion,
    trajectory,
    stack,
    pageErrors,
    consoleErrors,
    failures,
  };

  if (reportPath) {
    await mkdir(dirname(reportPath), { recursive: true });
    await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`);
  }

  if (cameraFrameDir) {
    await mkdir(cameraFrameDir, { recursive: true });
    for (const [key, value] of Object.entries(exportedCameraFrames?.latest ?? {})) {
      if (key === 'at' || typeof value !== 'string') continue;
      const decoded = dataUrlToBytes(value);
      if (!decoded) continue;
      await writeFile(
        join(cameraFrameDir, `${key}.${extensionForMimeType(decoded.mimeType)}`),
        decoded.bytes
      );
    }
    for (const [index, frames] of (exportedCameraFrames?.history ?? []).entries()) {
      const frameIndex = String(index).padStart(3, '0');
      for (const [key, value] of Object.entries(frames)) {
        if (key === 'at' || typeof value !== 'string') continue;
        const decoded = dataUrlToBytes(value);
        if (!decoded) continue;
        await writeFile(
          join(cameraFrameDir, `${frameIndex}-${key}.${extensionForMimeType(decoded.mimeType)}`),
          decoded.bytes
        );
      }
    }
  }

  console.log(JSON.stringify(report, null, 2));
  if (failures.length > 0) process.exitCode = 1;
} finally {
  await browser.close();
}
