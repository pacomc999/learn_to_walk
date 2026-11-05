// The 3D viewer: shows a creature, its world and recorded walks (replays from the studio) with three.js.
// The simulation's z axis points up, so the viewer works with z up too.

import * as THREE from "three";
import { OrbitControls } from "../vendor/OrbitControls.js";
import { h } from "./ui.js";
import { icon, STYLE_COLOURS } from "./icons.js";

const SPEEDS = [0.25, 0.5, 1, 2];

export class Viewer {
  constructor(container, { playbar = true, follow = true, autoplay = true, loop = true, pace = false } = {}) {
    this.container = container;
    this.container.classList.add("viewer");
    this.options = { playbar, autoplay, loop, pace };
    this.follow = follow;
    this.replay = null;
    this.time = 0;
    this.speed = 1;
    this.playing = false;
    this.disposed = false;
    this.highlightTest = null;
    this.bodyGroups = [];
    this.meshes = [];

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    container.append(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.01, 200);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(1.2, -1.8, 0.8);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.maxPolarAngle = Math.PI * 0.495;
    this.controls.minDistance = 0.2;
    this.controls.maxDistance = 30;
    this.controls.target.set(0, 0, 0.2);

    this.hemi = new THREE.HemisphereLight(0xffffff, 0x8a8f99, 1.1);
    this.scene.add(this.hemi);
    this.sun = new THREE.DirectionalLight(0xffffff, 2.2);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    this.sun.shadow.bias = -0.0004;
    this.sun.shadow.normalBias = 0.01;
    const box = this.sun.shadow.camera;
    box.left = box.bottom = -2.5; box.right = box.top = 2.5; box.near = 0.1; box.far = 20;
    this.scene.add(this.sun, this.sun.target);

    this.world = new THREE.Group();
    this.walker = new THREE.Group();
    this.scene.add(this.world, this.walker);

    this.overlay = h("div", { class: "viewer-overlay" });
    this.tools = h("div", { class: "viewer-tools" });
    this.message = h("div", { class: "viewer-message", hidden: true });
    this.loading = h("div", { class: "loading-bar", hidden: true });
    container.append(this.overlay, this.tools, this.message, this.loading);
    this._buildTools();
    if (playbar) this._buildPlaybar();

    // Clicking a part (without dragging the view) tells onPick its name
    this.onPick = null;
    this.onTime = null;  // called every frame with the playback time, for panels that follow the replay
    let downAt = null;
    this.renderer.domElement.addEventListener("pointerdown", (event) => { downAt = [event.clientX, event.clientY]; });
    this.renderer.domElement.addEventListener("pointerup", (event) => {
      if (!downAt || !this.onPick || Math.hypot(event.clientX - downAt[0], event.clientY - downAt[1]) > 4) return;
      const box = this.renderer.domElement.getBoundingClientRect();
      const pointer = new THREE.Vector2(((event.clientX - box.left) / box.width) * 2 - 1, -((event.clientY - box.top) / box.height) * 2 + 1);
      const ray = new THREE.Raycaster();
      ray.setFromCamera(pointer, this.camera);
      const hit = ray.intersectObjects(this.meshes, false)[0];
      this.onPick(hit ? hit.object.userData.name : null);
    });

    this.resizeObserver = new ResizeObserver(() => this._resize());
    this.resizeObserver.observe(container);
    this._resize();
    this.clock = new THREE.Clock();
    this._loop = this._loop.bind(this);
    requestAnimationFrame(this._loop);
  }

  // ---------- Building the scene ----------

  setLoading(on) { this.loading.hidden = !on; }

  showMessage(text) {
    this.message.hidden = !text;
    this.message.textContent = text || "";
  }

  setChips(chips) {
    this.overlay.replaceChildren(...chips.map(([key, value]) => h("div", { class: "chip" }, h("span", { class: "k" }, key), h("b", {}, value))));
  }

  banner(text, kind = "good", seconds = 4) {
    if (this.bannerElement) this.bannerElement.remove();
    if (!text) return;
    this.bannerElement = h("div", { class: `viewer-banner ${kind}` }, text);
    this.container.append(this.bannerElement);
    if (seconds) setTimeout(() => this.bannerElement && this.bannerElement.remove(), seconds * 1000);
  }

  // Show a replay. keepCamera keeps the viewpoint (for live edits of the same creature)
  load(replay, { keepCamera = false } = {}) {
    const sameWorld = this.replay && this.replay.ground === replay.ground && this.replay.style === replay.style;
    this.replay = replay;
    this.time = 0;
    this.showMessage("");
    this._buildWalker(replay);
    if (!keepCamera || !sameWorld) this._buildWorld(replay);
    this._buildPace(replay);
    this._buildBalance(replay);
    this._applyFrame(0);
    if (!keepCamera) this.frameCamera();
    this.playing = this.options.autoplay && replay.frames.length > 1;
    this._paintPlaybar();
  }

  _clearGroup(group) {
    for (const child of [...group.children]) {
      child.traverse((node) => {
        if (node.geometry) node.geometry.dispose();
        if (node.material) {
          if (node.material.map) node.material.map.dispose();
          node.material.dispose();
        }
      });
      group.remove(child);
    }
  }

  _buildWalker(replay) {
    this._clearGroup(this.walker);
    this.bodyGroups = replay.bodies.map(() => new THREE.Group());
    this.bodyGroups.forEach((group) => this.walker.add(group));
    this.meshes = [];
    for (const shape of replay.shapes) {
      const mesh = makeShape(shape);
      if (!mesh) continue;
      mesh.userData.name = shape.name || "";
      mesh.userData.baseColour = mesh.material.color.clone();
      this.bodyGroups[shape.body].add(mesh);
      this.meshes.push(mesh);
    }
    this._paintHighlight();
  }

  _buildWorld(replay) {
    this._clearGroup(this.world);
    const style = STYLE_COLOURS[replay.style] || STYLE_COLOURS.meadow;
    const dark = replay.style === "moon" || replay.style === "night";
    this.container.style.background = `linear-gradient(180deg, ${style.sky[0]}, ${style.sky[1]})`;
    this.scene.fog = new THREE.Fog(new THREE.Color(style.sky[1]), 8, 40);
    this.hemi.intensity = dark ? 0.55 : 1.1;
    this.sun.intensity = dark ? 1.4 : 2.2;
    this.sun.color = new THREE.Color(dark ? 0xc8d4ff : 0xfff6e8);
    const size = replay.size || 1;

    // The endless plane around everything
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(400, 400), new THREE.MeshStandardMaterial({
      color: new THREE.Color(style.ground), map: groundTexture(style, replay.style), roughness: 0.95,
    }));
    plane.material.map.repeat.set(400 / (0.5 * Math.max(size, 0.5)), 400 / (0.5 * Math.max(size, 0.5)));
    plane.receiveShadow = true;
    const patches = this._patches(replay);
    plane.position.z = patches.length ? -0.0008 : 0;
    this.world.add(plane);
    for (const patch of patches) {
      const texture = plane.material.map.clone();
      texture.repeat.set(1, 1);
      texture.needsUpdate = true;
      this.world.add(heightsMesh(patch, style, texture, 0.5 * Math.max(size, 0.5)));
    }
    // Name labels: fixed in place, or floating above a body (racers) and following it
    this.labels = [];
    for (const label of replay.labels || []) {
      const sprite = textSprite(label.text, dark ? "#ffffff" : "#111827", 0.95);
      sprite.scale.multiplyScalar(2.2);
      sprite.material.depthTest = false;  // always readable, never hidden behind a walker
      sprite.renderOrder = 10;
      if (label.body !== undefined) this.labels.push({ sprite, body: label.body, lift: label.z });
      else sprite.position.set(label.x, label.y, label.z);
      this.world.add(sprite);
    }
    this._addMetreMarks(replay, style);
    if (dark) this.world.add(starField());
  }

  // The height maps of a replay: one, or one per lane in a race
  _patches(replay) {
    const ground = replay.ground || {};
    if (ground.kind === "heights") return [ground];
    if (ground.kind === "lanes") return ground.patches.filter((patch) => patch.kind === "heights");
    return [];
  }

  _addMetreMarks(replay, style) {
    if (replay.style === "studio") return;  // the plain studio floor of the creature preview has no track
    const size = Math.max(replay.ground && replay.ground.kind === "lanes" ? 1 : replay.size || 1, 0.6);
    const xs = replay.frames.length ? replay.frames.map((frame) => frame[0]) : [0];
    const last = replay.marks_to ? Math.ceil(replay.marks_to) : Math.ceil(Math.max(...xs, 2) + 3);
    const lanes = replay.ground && replay.ground.kind === "lanes" ? replay.ground : null;
    const side = lanes ? lanes.count * lanes.lane / 2 + 0.15 : 0.6 * size;
    for (let metre = 0; metre <= last; metre++) {
      const post = new THREE.Mesh(new THREE.CylinderGeometry(0.008 * size, 0.008 * size, 0.12 * size, 8),
        new THREE.MeshStandardMaterial({ color: metre === 0 ? 0x4f46e5 : 0xffffff, roughness: 0.6 }));
      post.rotation.x = Math.PI / 2;
      post.position.set(metre, side, 0.06 * size + this._groundHeight(replay, metre, side));
      post.castShadow = true;
      this.world.add(post);
      if (metre % 1 === 0) {
        const label = textSprite(metre === 0 ? "start" : `${metre} m`, replay.style === "moon" || replay.style === "night" ? "#ffffff" : "#1f2937");
        label.position.set(metre, side, 0.16 * size + this._groundHeight(replay, metre, side));
        label.scale.multiplyScalar(size);
        this.world.add(label);
      }
    }
  }

  _groundHeight(replay, x, y) {
    const ground = this._patches(replay).find((patch) => y >= patch.y0 && y <= patch.y0 + patch.step * (patch.rows - 1)) || this._patches(replay)[0];
    if (!ground) return 0;
    const column = Math.round((x - ground.x0) / ground.step);
    const row = Math.round((y - ground.y0) / ground.step);
    if (column < 0 || row < 0 || column >= ground.cols || row >= ground.rows) return 0;
    return ground.data[row * ground.cols + column] / 10000;
  }

  // In the creature preview: the centre of mass (a ball with a line down to the ground) and the area the feet
  // stand on. Over the area, it can stand; outside it, it tips over
  _buildBalance(replay) {
    if (this.balance) { this.scene.remove(this.balance); this.balance.traverse((n) => { if (n.geometry) n.geometry.dispose(); }); this.balance = null; }
    if (!replay.centre_of_mass) return;
    const group = new THREE.Group();
    const [x, y, z] = replay.centre_of_mass;
    const size = Math.max(replay.size || 1, 0.5);
    const inside = !replay.support || replay.support.length < 3 || pointInPolygon([x, y], replay.support);
    const colour = inside ? 0x10b981 : 0xef4444;
    const ball = new THREE.Mesh(new THREE.SphereGeometry(0.014 * size, 20, 12), new THREE.MeshBasicMaterial({ color: colour, depthTest: false }));
    ball.position.set(x, y, z);
    ball.renderOrder = 5;
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(x, y, z), new THREE.Vector3(x, y, 0.001)]),
      new THREE.LineDashedMaterial({ color: colour, dashSize: 0.01, gapSize: 0.008, depthTest: false }));
    line.computeLineDistances();
    line.renderOrder = 5;
    group.add(ball, line);
    if (replay.support && replay.support.length >= 3) {
      const shape = new THREE.Shape(replay.support.map(([px, py]) => new THREE.Vector2(px, py)));
      const area = new THREE.Mesh(new THREE.ShapeGeometry(shape), new THREE.MeshBasicMaterial({ color: colour, transparent: true, opacity: 0.18, depthWrite: false }));
      area.position.z = 0.001;
      const outline = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(replay.support.map(([px, py]) => new THREE.Vector3(px, py, 0.0015))),
        new THREE.LineBasicMaterial({ color: colour }));
      group.add(area, outline);
    }
    this.balance = group;
    this.balance.visible = this.showBalance !== false;
    this.scene.add(group);
  }

  // A soft marker that moves at the asked speed, so you can see if the walker keeps up
  _buildPace(replay) {
    if (this.paceMarker) { this.scene.remove(this.paceMarker); this.paceMarker = null; }
    if (!this.options.pace || !replay.stats || !replay.stats.target_speed || replay.stats.turn_rate) return;
    const size = Math.max(replay.size || 1, 0.6);
    const marker = new THREE.Mesh(new THREE.RingGeometry(0.06 * size, 0.08 * size, 40),
      new THREE.MeshBasicMaterial({ color: 0x6366f1, transparent: true, opacity: 0.75, side: THREE.DoubleSide, depthWrite: false }));
    marker.renderOrder = 2;
    this.paceMarker = marker;
    this.paceStart = replay.frames[0][0];
    this.scene.add(marker);
  }

  frameCamera() {
    const replay = this.replay;
    const size = Math.max(replay ? replay.size || 1 : 1, 0.5);
    const height = replay ? replay.stand_height || 0.25 : 0.25;
    let x = replay && replay.frames.length ? replay.frames[0][0] : 0;
    if (this.live && this.bodyGroups.length) x = this.bodyGroups[0].position.x;
    if (replay && replay.torsos) {
      this._applyFrame(this.time);
      const middle = this._torsoPosition();
      this.controls.target.set(middle.x, 0, height * 0.85);
      this.camera.position.set(middle.x + 0.9 * size, -1.6 * size, height + 0.8 * size);
      this.controls.update();
      return;
    }
    this.controls.target.set(x, 0, height * 0.85);
    this.camera.position.set(x + 0.75 * size, -1.45 * size, height + 0.45 * size);
    this.controls.update();
  }

  // ---------- Highlighting parts ----------

  // test(name) says which shapes to highlight; null clears it
  highlight(test) {
    this.highlightTest = test;
    this._paintHighlight();
  }

  _paintHighlight() {
    for (const mesh of this.meshes) {
      const on = this.highlightTest && this.highlightTest(mesh.userData.name);
      mesh.material.emissive = new THREE.Color(on ? 0x6366f1 : 0x000000);
      mesh.material.emissiveIntensity = on ? 0.3 : 0;
    }
  }

  // ---------- Playing ----------

  // ---------- Live mode: frames arrive while it plays ----------

  startLive(header) {
    this.options.loop = false;
    this.load(header);
    this.live = true;
    this.playing = true;
    if (this.playbar) this.playbar.hidden = true;
  }

  pushLive(frames) {
    if (!this.replay || !frames.length) return;
    this.replay.frames.push(...frames);
    const keep = 750;
    if (this.replay.frames.length > keep) {
      const drop = this.replay.frames.length - keep;
      this.replay.frames.splice(0, drop);
      this.time = Math.max(0, this.time - drop / this.replay.fps);
    }
  }

  get duration() {
    return this.replay ? (this.replay.frames.length - 1) / this.replay.fps : 0;
  }

  play() { if (this.replay && this.replay.frames.length > 1) { if (this.time >= this.duration) this.time = 0; this.playing = true; this._paintPlaybar(); } }
  pause() { this.playing = false; this._paintPlaybar(); }
  seek(seconds) { this.time = Math.max(0, Math.min(this.duration, seconds)); this._applyFrame(this.time); this._paintPlaybar(); }

  _applyFrame(seconds) {
    const replay = this.replay;
    if (!replay || !replay.frames.length) return;
    const position = Math.min(seconds * replay.fps, replay.frames.length - 1);
    const index = Math.floor(position);
    const blend = position - index;
    const a = replay.frames[index];
    const b = replay.frames[Math.min(index + 1, replay.frames.length - 1)];
    const qa = new THREE.Quaternion(), qb = new THREE.Quaternion();
    this.bodyGroups.forEach((group, body) => {
      const o = body * 7;
      group.position.set(
        a[o] + (b[o] - a[o]) * blend, a[o + 1] + (b[o + 1] - a[o + 1]) * blend, a[o + 2] + (b[o + 2] - a[o + 2]) * blend);
      qa.set(a[o + 4], a[o + 5], a[o + 6], a[o + 3]);
      qb.set(b[o + 4], b[o + 5], b[o + 6], b[o + 3]);
      group.quaternion.copy(qa.slerp(qb, blend));
    });
    for (const label of this.labels || []) {
      const body = this.bodyGroups[label.body];
      if (body) label.sprite.position.set(body.position.x, body.position.y, label.lift);
    }
    if (this.paceMarker) {
      const x = this.paceStart + replay.stats.target_speed * Math.min(seconds, replay.stats.seconds + 1);
      this.paceMarker.position.set(x, 0, this._groundHeight(replay, x, 0) + 0.004);
    }
  }

  // What the camera follows: the torso, or the middle of all the racers
  _torsoPosition() {
    if (!this.bodyGroups.length) return new THREE.Vector3();
    const torsos = this.replay && this.replay.torsos;
    if (!torsos) return this.bodyGroups[0].position;
    const middle = new THREE.Vector3();
    for (const index of torsos) middle.add(this.bodyGroups[index].position);
    return middle.divideScalar(torsos.length);
  }

  _loop() {
    if (this.disposed) return;
    requestAnimationFrame(this._loop);
    const delta = Math.min(this.clock.getDelta(), 0.1);
    if (this.playing && this.replay) {
      if (this.live) {
        // Stay a little behind the newest frame, and jump ahead if the page fell far behind
        this.time = Math.min(this.time + delta, this.duration);
        if (this.time < this.duration - 0.6) this.time = this.duration - 0.15;
      } else {
        this.time += delta * this.speed;
        if (this.time >= this.duration) {
          if (this.options.loop) this.time = 0;
          else { this.time = this.duration; this.playing = false; }
        }
      }
      const before = this._torsoPosition().clone();
      this._applyFrame(this.time);
      if (this.follow) {
        const after = this._torsoPosition();
        const shift = new THREE.Vector3(after.x - before.x, after.y - before.y, 0);
        if (shift.length() < 1) {  // a jump back to the start moves the camera with the loop
          this.camera.position.add(shift);
          this.controls.target.add(shift);
        } else {
          this.frameCamera();
        }
      }
      this._paintPlaybar(true);
    }
    const torso = this._torsoPosition();
    this.sun.position.set(torso.x + 2, torso.y - 2.5, torso.z + 5);
    this.sun.target.position.set(torso.x, torso.y, 0);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    if (this.onTime && this.replay) this.onTime(this.time);
  }

  _resize() {
    const width = this.container.clientWidth || 300;
    const height = this.container.clientHeight || 300;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  // ---------- Buttons ----------

  _buildTools() {
    const follow = h("button", { class: `vbtn ${this.follow ? "on" : ""}`, title: "Follow the walker", html: icon.follow });
    follow.addEventListener("click", () => { this.follow = !this.follow; follow.classList.toggle("on", this.follow); });
    const reset = h("button", { class: "vbtn", title: "Reset the view", html: icon.camera, onclick: () => this.frameCamera() });
    const full = h("button", { class: "vbtn", title: "Full screen", html: icon.expand, onclick: () => {
      if (document.fullscreenElement) document.exitFullscreen(); else this.container.requestFullscreen();
    } });
    const picture = h("button", { class: "vbtn", title: "Save a picture", html: icon.image, onclick: () => this.savePicture() });
    this.recordButton = h("button", { class: "vbtn", title: "Record a video of the 3D view", html: icon.record, onclick: () => this.toggleRecording() });
    this.recordButton.style.color = "#dc2626";
    this.tools.append(follow, reset, picture, this.recordButton, full);
  }

  // ---------- Pictures and videos ----------

  savePicture() {
    download(this.snapshot(), "walk-studio.png");
  }

  // Record the 3D view (and its playback) as a WebM video; a second press stops and saves it
  toggleRecording() {
    if (this.recorder) { this.recorder.stop(); return; }
    if (!window.MediaRecorder || !this.renderer.domElement.captureStream) {
      this.banner("This browser cannot record videos.", "bad");
      return;
    }
    const stream = this.renderer.domElement.captureStream(30);
    const type = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"].find((t) => MediaRecorder.isTypeSupported(t)) || "";
    const chunks = [];
    this.recorder = new MediaRecorder(stream, type ? { mimeType: type, videoBitsPerSecond: 8e6 } : {});
    this.recorder.ondataavailable = (event) => { if (event.data.size) chunks.push(event.data); };
    this.recorder.onstop = () => {
      const blob = new Blob(chunks, { type: "video/webm" });
      download(URL.createObjectURL(blob), "walk-studio.webm");
      this.recorder = null;
      this.recordButton.classList.remove("recording");
      this.banner("Video saved to your downloads.", "good", 3);
    };
    this.recorder.start(250);
    this.recordButton.classList.add("recording");
    this.banner("Recording. Press the red button again to stop and save.", "good", 4);
    if (this.replay && this.replay.frames.length > 1) { this.seek(0); this.play(); }
  }

  _buildPlaybar() {
    this.playButton = h("button", { html: icon.play, title: "Play or pause", onclick: () => (this.playing ? this.pause() : this.play()) });
    this.scrub = h("input", { type: "range", min: 0, max: 1, step: 0.001, value: 0 });
    this.scrub.addEventListener("input", () => { this.pause(); this.seek(Number(this.scrub.value) * this.duration); });
    this.timeText = h("span", { class: "time" }, "0.0 s");
    this.speedButton = h("button", { class: "speed", title: "Playback speed", onclick: () => {
      this.speed = SPEEDS[(SPEEDS.indexOf(this.speed) + 1) % SPEEDS.length];
      this._paintPlaybar();
    } }, "1x");
    const restart = h("button", { html: icon.restart, title: "From the start", onclick: () => { this.seek(0); this.play(); if (this.follow) this.frameCamera(); } });
    this.playbar = h("div", { class: "playbar" }, this.playButton, restart, this.scrub, this.timeText, this.speedButton);
    this.container.append(this.playbar);
  }

  _paintPlaybar(onlyTime = false) {
    if (!this.playbar) return;
    if (this.live) { this.playbar.hidden = true; return; }
    const share = this.duration ? this.time / this.duration : 0;
    this.scrub.value = share;
    this.scrub.style.setProperty("--fill", `${share * 100}%`);
    this.timeText.textContent = `${this.time.toFixed(1)} / ${this.duration.toFixed(1)} s`;
    if (onlyTime) return;
    this.playButton.innerHTML = this.playing ? icon.pause : icon.play;
    this.speedButton.textContent = `${this.speed}x`;
    this.playbar.hidden = !this.replay || this.replay.frames.length <= 1;
  }

  snapshot() {
    this.renderer.render(this.scene, this.camera);
    return this.renderer.domElement.toDataURL("image/png");
  }

  // A small JPEG of the view as a data address, for saved library items
  thumbnailUrl(width = 320, height = 200) {
    return this._thumbnailCanvas(width, height).toDataURL("image/jpeg", 0.8);
  }

  _thumbnailCanvas(width, height) {
    this.renderer.render(this.scene, this.camera);
    const source = this.renderer.domElement;
    const canvas = document.createElement("canvas");
    canvas.width = width; canvas.height = height;
    const context = canvas.getContext("2d");
    const scale = Math.max(width / source.width, height / source.height);
    const w = source.width * scale, h = source.height * scale;
    context.fillStyle = "#dfe7ef";
    context.fillRect(0, 0, width, height);
    context.drawImage(source, (width - w) / 2, (height - h) / 2, w, h);
    return canvas;
  }

  // A small JPEG of the view, framed on the walker, for cards (resolves to a Blob)
  thumbnail(width = 480, height = 300) {
    const canvas = this._thumbnailCanvas(width, height);
    return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.82));
  }

  dispose() {
    if (this.recorder) this.recorder.stop();
    this.disposed = true;
    this.resizeObserver.disconnect();
    this._clearGroup(this.walker);
    this._clearGroup(this.world);
    this.controls.dispose();
    this.renderer.dispose();
    this.renderer.forceContextLoss();
    this.container.replaceChildren();
  }
}

function pointInPolygon([x, y], polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i], [xj, yj] = polygon[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function download(url, name) {
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  document.body.append(link);
  link.click();
  link.remove();
}

// ---------- Shapes ----------

function makeShape(shape) {
  const [r, g, b] = shape.rgba;
  const material = new THREE.MeshStandardMaterial({ color: new THREE.Color(r, g, b).convertSRGBToLinear(), roughness: 0.55, metalness: 0.02 });
  const s = shape.size;
  let geometry;
  if (shape.type === "sphere") geometry = new THREE.SphereGeometry(s[0], 32, 20);
  else if (shape.type === "ellipsoid") { geometry = new THREE.SphereGeometry(1, 40, 24); geometry.scale(s[0], s[1], s[2]); }
  else if (shape.type === "capsule") { geometry = new THREE.CapsuleGeometry(s[0], 2 * s[1], 8, 20); geometry.rotateX(Math.PI / 2); }
  else if (shape.type === "cylinder") { geometry = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 28); geometry.rotateX(Math.PI / 2); }
  else if (shape.type === "box") geometry = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
  else return null;
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(...shape.pos);
  mesh.quaternion.set(shape.quat[1], shape.quat[2], shape.quat[3], shape.quat[0]);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

// ---------- Ground ----------

function groundTexture(style, styleName) {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 128;
  const context = canvas.getContext("2d");
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, 128, 128);
  if (styleName === "street") {
    context.fillStyle = "#e9e9e9";
    for (let y = 0; y < 128; y += 32) for (let x = (y / 32) % 2 ? 16 : 0; x < 128; x += 32) context.fillRect(x + 1, y + 1, 30, 30);
  } else if (styleName === "snow") {
    context.fillStyle = "#f2f7fb";
    context.fillRect(0, 0, 64, 64); context.fillRect(64, 64, 64, 64);
  } else {
    context.fillStyle = "#eeeeee";
    context.fillRect(0, 0, 64, 64); context.fillRect(64, 64, 64, 64);
    for (let i = 0; i < 300; i++) {
      context.fillStyle = `rgba(0,0,0,${Math.random() * 0.05})`;
      context.fillRect(Math.random() * 128, Math.random() * 128, 2, 2);
    }
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  return texture;
}

function heightsMesh(ground, style, texture, tile) {
  const { rows, cols, step, x0, y0, data } = ground;
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(rows * cols * 3);
  const colours = new Float32Array(rows * cols * 3);
  const uvs = new Float32Array(rows * cols * 2);
  let top = 1;
  for (const value of data) top = Math.max(top, value);
  const base = new THREE.Color(style.ground).convertSRGBToLinear();
  const low = new THREE.Color(style.dark).convertSRGBToLinear();
  const mixed = new THREE.Color();
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c;
      const z = data[i] / 10000;
      positions.set([x0 + c * step, y0 + r * step, z], i * 3);
      uvs.set([(x0 + c * step) / tile, (y0 + r * step) / tile], i * 2);
      mixed.copy(low).lerp(base, Math.min(1, 0.85 + 0.15 * data[i] / top));
      colours.set([mixed.r, mixed.g, mixed.b], i * 3);
    }
  }
  const index = new Uint32Array((rows - 1) * (cols - 1) * 6);
  let k = 0;
  for (let r = 0; r < rows - 1; r++) {
    for (let c = 0; c < cols - 1; c++) {
      const a = r * cols + c, b = a + 1, d = a + cols, e = d + 1;
      index.set([a, b, e, a, e, d], k);
      k += 6;
    }
  }
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colours, 3));
  geometry.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
  geometry.setIndex(new THREE.BufferAttribute(index, 1));
  geometry.computeVertexNormals();
  const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ vertexColors: true, map: texture, roughness: 0.95 }));
  mesh.receiveShadow = true;
  return mesh;
}

// Stars on a dome far around the scene, for night and Moon skies
function starField() {
  const positions = [];
  for (let i = 0; i < 900; i++) {
    const angle = Math.random() * Math.PI * 2;
    const height = 0.05 + Math.random() * 0.9;
    const radius = 70;
    positions.push(Math.cos(angle) * radius * Math.sqrt(1 - height * height), Math.sin(angle) * radius * Math.sqrt(1 - height * height), height * radius);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  const stars = new THREE.Points(geometry, new THREE.PointsMaterial({ color: 0xffffff, size: 0.25, sizeAttenuation: true, fog: false, transparent: true, opacity: 0.85 }));
  stars.userData.followCamera = true;
  return stars;
}

function textSprite(text, colour, alpha = 0.75) {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  const font = "600 36px Inter, system-ui, sans-serif";
  context.font = font;
  const width = Math.min(1024, Math.ceil(context.measureText(text).width) + 24);
  canvas.width = width; canvas.height = 64;
  context.font = font;
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillStyle = colour;
  context.globalAlpha = alpha;
  context.fillText(text, width / 2, 32);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthWrite: false, transparent: true }));
  sprite.scale.set(0.06 * width / 64, 0.06, 1);
  return sprite;
}
