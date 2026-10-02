// Compatibility entry point: full 60-frame acceptance in the redesigned UI.
process.env.TRACKER_TEST_FRAMES ||= '60';
await import('./ui_redesign_smoke.mjs');
