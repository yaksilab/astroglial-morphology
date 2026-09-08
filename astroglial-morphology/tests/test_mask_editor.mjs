// Exercise the actual renderer's pointer events and serialized save payload.
// The minimal DOM/canvas supplies rendering sinks; editing code is unmodified.
// Run with: node tests/test_mask_editor.mjs (also supports node --test).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const source = readFileSync(new URL('../src/astroglial_morphology/gui/components/mask_editor/editor.js', import.meta.url), 'utf8');
const { default: render } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
globalThis.Image = class {};

function editor(initial = new Int32Array(100 * 100)) {
  const elements = new Map();
  const context = {
    clearRect() {}, drawImage() {}, putImageData() {},
    createImageData(w, h) { return { data: new Uint8ClampedArray(w * h * 4) }; },
    beginPath() {}, moveTo() {}, lineTo() {}, stroke() {},
  };
  function element(key) {
    if (!elements.has(key)) elements.set(key, {
      style: {}, checked: false, classList: { toggle() {} },
      listeners: new Map(), setAttribute() {}, setPointerCapture() {},
      addEventListener(name, fn) { this.listeners.set(name, fn); },
      removeEventListener(name) { this.listeners.delete(name); },
      getContext() { return context; },
      getBoundingClientRect() { return { left: 0, top: 0, width: 100, height: 100 }; },
    });
    return elements.get(key);
  }
  const tools = ['select', 'brush', 'erase', 'split', 'pan'].map(tool => {
    const button = element(`tool:${tool}`);
    button.dataset = { tool };
    return button;
  });
  const root = element('root');
  root.querySelector = selector => {
    const role = selector.match(/data-role="([^"]+)"/);
    return element(role ? role[1] : selector);
  };
  root.querySelectorAll = () => tools;
  let saved;
  render({ parentElement: root,
    data: { width: 100, height: 100, max_label: Math.max(...initial),
      masks_b64: Buffer.from(initial.buffer).toString('base64') },
    setTriggerValue(name, payload) { assert.equal(name, 'save'); saved = payload; },
  });
  function pointer(name, x, y, shiftKey = false) {
    element('overlay').listeners.get(name)({ clientX: x, clientY: y, pointerId: 1, shiftKey });
  }
  function select(x, y, shiftKey = false) {
    tools.find(b => b.dataset.tool === 'select').onclick();
    pointer('pointerdown', x, y, shiftKey);
    pointer('pointerup', x, y, shiftKey);
  }
  return {
    element, select,
    draw(x, y) {
      tools.find(b => b.dataset.tool === 'brush').onclick();
      pointer('pointerdown', x, y);
      pointer('pointermove', x + 10, y);
      pointer('pointermove', x + 10, y + 10);
      pointer('pointermove', x, y + 10);
      pointer('pointerup', x, y);
    },
    extendSelected() {
      element('new-label').checked = false;
      element('new-label').onchange();
    },
    save() {
      element('commit').onclick();
      const bytes = Buffer.from(saved.masks_b64, 'base64');
      return Int32Array.from({ length: bytes.length / 4 }, (_, i) => bytes.readInt32LE(i * 4));
    },
  };
}

test('three separate outlines keep distinct cell IDs through save and reload', () => {
  const gui = editor();
  gui.draw(10, 10); gui.draw(40, 40); gui.draw(70, 70);
  const saved = gui.save();
  assert.equal(new Set([saved[15 * 100 + 15], saved[45 * 100 + 45], saved[75 * 100 + 75]]).size, 3);
  assert.deepEqual([...new Set(saved)].sort(), [0, 1, 2, 3]);
  const reloaded = editor(saved);
  reloaded.draw(70, 10);
  assert.equal(reloaded.save()[15 * 100 + 75], 4);
});

test('extending a cell explicitly retains its selected label', () => {
  const gui = editor();
  gui.draw(10, 10);
  gui.extendSelected();
  gui.draw(18, 10);
  assert.deepEqual([...new Set(gui.save())].sort(), [0, 1]);
});

test('extension without exactly one selected cell makes no edit', () => {
  const gui = editor();
  gui.extendSelected(); gui.draw(10, 10);
  assert.match(gui.element('info').textContent, /Select exactly one cell/);
  assert.ok(gui.save().every(label => label === 0));
});

test('extension with several selected cells makes no edit', () => {
  const gui = editor();
  gui.draw(10, 10); gui.draw(40, 40);
  gui.select(15, 15); gui.select(45, 45, true);
  const before = gui.save();
  gui.extendSelected(); gui.draw(70, 70);
  assert.deepEqual(gui.save(), before);
});

test('separate regions preserves all pixels, other cell IDs and undo/redo', () => {
  const initial = new Int32Array(100 * 100);
  for (const [x, y, size, label] of [[10, 10, 10, 5], [40, 40, 8, 5], [70, 70, 6, 5], [70, 10, 5, 9]]) {
    for (let dy = 0; dy < size; dy++) for (let dx = 0; dx < size; dx++) {
      initial[(y + dy) * 100 + x + dx] = label;
    }
  }
  const gui = editor(initial);
  gui.select(12, 12);
  gui.element('separate').onclick();
  const separated = gui.save();
  assert.equal(separated[12 * 100 + 12], 5);
  assert.equal(separated[42 * 100 + 42], 10);
  assert.equal(separated[72 * 100 + 72], 11);
  for (let i = 0; i < initial.length; i++) {
    assert.equal(separated[i] !== 0, initial[i] !== 0);
    if (initial[i] === 9) assert.equal(separated[i], 9);
  }
  gui.element('undo').onclick();
  assert.deepEqual(gui.save(), initial);
  gui.element('redo').onclick();
  assert.deepEqual(gui.save(), separated);
});

test('diagonal contact stays connected, while opposite row edges do not', () => {
  const initial = new Int32Array(100 * 100);
  initial[10 * 100 + 10] = initial[11 * 100 + 11] = 5;
  initial[99] = initial[100] = 7;
  const gui = editor(initial);
  gui.select(10, 10); gui.element('separate').onclick();
  assert.deepEqual(gui.save(), initial);
  gui.select(99, 0); gui.element('separate').onclick();
  const result = gui.save();
  assert.notEqual(result[99], result[100]);
  assert.equal(result[1010], result[1111]);
});
