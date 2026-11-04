// How it works: the ideas behind the studio, in plain words.

import { h } from "../ui.js";

const SECTIONS = [
  ["The short version", [
    "A creature in Walk Studio is a body in a physics simulation (MuJoCo): solid parts joined by hinges, with a motor at every hinge. Gravity, friction and momentum work as they do in the real world.",
    "Its brain is a small neural network. Fifty times a second it receives what the creature senses (its tilt, its speed, the angles of its joints, which feet touch the ground, the speed it is asked to walk at) and answers with a command for every motor.",
    "At first the brain is random, so the creature flails and falls. Learning nudges the brain, a tiny bit at a time, toward the moves that earned more points. After a few million moments of practice, walking appears by itself. Nobody programs the steps.",
  ]],
  ["Points: what it cares about", [
    "Every moment the walker earns or loses points: most for moving forward at exactly the asked speed, a little for staying on its feet, minus a little for every joule of energy its muscles use, and minus a little for drifting off the straight line.",
    "These points are the only thing the walker learns from, so they shape its style. Make energy expensive and it walks calmly (or gives up). Make staying up too valuable and it learns to stand still. The World step lets you change all of them under \"What it cares about\".",
  ]],
  ["Practice, lessons and exams", [
    "Training runs several copies of the world at once, one per processor core. After each round of practice the brain studies what happened (a lesson) and changes slightly. One step is a fiftieth of a second of practice in one world, so a million steps is about five and a half hours of practice, done much faster than real time.",
    "Practice includes random tries, which is how it discovers new moves, so practice scores are noisy. Every so often it takes an exam: the same attempts each time, at speeds spread over the asked range, with no randomness. The exam decides which brain is the best so far, and every exam is recorded as a 3D replay, so you can watch it grow up.",
    "\"How well it walks\" is the share of the asked distance it covered in the exam. 100% means it kept up with the asked speed for the whole attempt without falling.",
  ]],
  ["Muscles: direct or springy", [
    "Direct muscles (the standard) push with exactly the force the brain asks for, and let the joint swing freely when it asks for none, like real muscles. This is how the newest learn_to_walk ducks move. With nothing to hold it up, a new creature collapses at first: the brain has to learn to carry its own weight.",
    "Springy muscles pull each joint toward an angle the brain picks, like a spring. They hold the body up from the very first moment, so learning starts easier, but the movements end up stiffer and more robotic.",
    "With direct muscles, resting a joint against its end stop holds the body up for free. Walkers love this trick and learn odd crouches. The world setting \"Penalty for joints at their limits\" (on by default) takes a few points away for it. Switch it off to see what they do without it, or raise it carefully: when the penalty costs more than walking earns, walkers learn to fall on purpose.",
  ]],
  ["Build your own", [
    "Choose \"Build your own\" on the Creature step. Start from a ready-made body (a four-legged table, a two-legged stick walker, a six-legged bug, a one-legged hopper) or from just a body, then attach parts: sticks, balls and plates. Click a part in the list or in the 3D view to change it.",
    "Each part hangs on its parent and is joined to it by a joint: fixed, swinging forward and back, swinging sideways, twisting, or a hip that swings both ways. Every moving joint gets a muscle. Turn on \"Mirrored pair\" and the part gets a twin on the other side, with everything that hangs on it. Mirrored creatures can use the symmetric brain; lopsided ones get a free brain.",
    "Mark the parts that touch the ground with \"Feels the ground\", so the brain knows when they land. Use the drop test to check that the muscles are strong enough to hold the body up.",
  ]],
  ["Designing the scoring", [
    "The World step lets you rewrite the rules of the game. Forward points can follow three shapes: exactly the asked speed (the most points at that speed, fewer when slower or faster), close to the asked speed (a gentler bell curve), or as fast as possible. Then come staying up, saving energy, walking straight and a cost for falling.",
    "Style rules add opinions about how it should move: keep the body level, stand tall, smooth moves, lift its feet, face forward, a steady ride and no leaning on joint limits. Each has a switch and a strength. Small strengths nudge; big ones can take over and teach odd habits.",
    "Scoring at a glance does the sums for you: what standing still, walking, a crouched walk and rushing are worth per second, and warnings when a trap opens up. During training, the chart \"Where its points come from\" shows which rules it is actually earning or paying.",
  ]],
  ["Watching, testing and racing", [
    "Every exam is recorded in 3D. On a walker's page you can scroll through them to watch it grow up, and the panel Inside the walk shows its gait (when each foot is on the ground) and what the brain tells every muscle.",
    "The Live tab lets it walk in real time while you change the asked speed (arrow keys up and down) and push it from the side (arrow keys left and right). The Test drive tab sends a trained walker into any world at any speed, and the Report card runs seven challenge ladders (speed, cobblestones, stairs, ice, a backpack, wind, other planets) and hands out medals. Race puts up to six walkers side by side. Every 3D view can save a picture or record a video.",
    "Your learn_to_walk ducks can come along: on My walkers, Bring over learn_to_walk ducks reads your catalogue and turns every duck with a kept brain into a walker here.",
  ]],
  ["The brain", [
    "The brain has layers of thinking cells. Each cell adds up what the layer before it says, with its own weights, and passes on a response. More layers and more cells can learn more complicated moves but need more practice. The standard brain (two layers of 256) taught the learn_to_walk ducks.",
    "The symmetric brain shares one way of thinking between the left and right side, mirrored. Animals are left/right symmetric, so this saves the brain from learning everything twice, and walking gaits with alternating legs appear much sooner.",
    "A second network, the judge, learns to guess how many points are coming. Learning uses the judge to tell which moves were better than expected.",
  ]],
  ["Tips", [
    "Start simple: the standard duck or puppy on the Flat Meadow learns to walk in roughly one to three hours on a laptop.",
    "Before a long training, run the drop test on the Creature step. A creature that collapses under its own weight needs stronger muscles or a lighter body.",
    "Hard worlds (stairs, hills, rocky ground) go better with \"Gradually harder\" switched on, with \"See the ground ahead\" in the brain's senses, or by starting from a brain that already walks on flat ground (\"Teach a new world\" on a walker's page).",
    "Training stops by itself when it stops getting better. You can pause and continue at any time; it carries on from where it stopped.",
    "The puppy has four legs, so it usually balances sooner than the duck, but it has more motors to coordinate.",
  ]],
];

export function render(main) {
  const page = h("div", { class: "page" });
  main.append(page);
  page.append(h("div", { class: "page-head" }, h("div", {}, h("div", { class: "eyebrow" }, "Guide"), h("h1", {}, "How it works"),
    h("div", { class: "sub" }, "What happens when a creature learns to walk, without any code."))));
  const guide = h("div", { class: "guide card card-pad", style: { padding: "8px 28px 28px" } });
  for (const [title, paragraphs] of SECTIONS) {
    guide.append(h("h2", {}, title));
    if (title === "Tips") guide.append(h("ul", {}, ...paragraphs.map((text) => h("li", {}, text))));
    else for (const text of paragraphs) guide.append(h("p", {}, text));
  }
  page.append(guide);
  return null;
}
