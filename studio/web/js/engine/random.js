// Seeded random numbers, so a terrain or an exam can be repeated exactly. Python used NumPy's generator;
// this is a different (but just as good) generator, so the exact numbers differ from the Python studio.

// sfc32: a small, fast generator with 128 bits of state, seeded through splitmix32
export class Random {
  constructor(seed = 0) {
    let s = (Number(seed) >>> 0) ^ 0x9e3779b9;
    const next = () => {  // splitmix32 spreads the seed over the four state words
      s = (s + 0x9e3779b9) >>> 0;
      let z = s;
      z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
      z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35);
      return (z ^ (z >>> 16)) >>> 0;
    };
    this.a = next(); this.b = next(); this.c = next(); this.d = next();
    for (let i = 0; i < 12; i++) this.random();
    this.spare = null;  // the second normal number from the last Box-Muller pair
  }

  // A number from 0 (included) to 1 (not included)
  random() {
    const t = (((this.a + this.b) >>> 0) + this.d) >>> 0;
    this.d = (this.d + 1) >>> 0;
    this.a = this.b ^ (this.b >>> 9);
    this.b = (this.c + (this.c << 3)) >>> 0;
    this.c = (this.c << 21) | (this.c >>> 11);
    this.c = (this.c + t) >>> 0;
    return t / 4294967296;
  }

  uniform(low = 0, high = 1) {
    return low + (high - low) * this.random();
  }

  // A whole number from low (included) to high (not included), as NumPy's integers()
  integers(low, high) {
    return low + Math.floor(this.random() * (high - low));
  }

  // A normal (bell curve) number with mean 0 and spread 1
  normal() {
    if (this.spare !== null) { const value = this.spare; this.spare = null; return value; }
    let u = 0;
    while (u === 0) u = this.random();
    const v = this.random();
    const radius = Math.sqrt(-2 * Math.log(u));
    this.spare = radius * Math.sin(2 * Math.PI * v);
    return radius * Math.cos(2 * Math.PI * v);
  }

  choice(items) {
    return items[this.integers(0, items.length)];
  }
}

// Python's round(): halves go to the even number (1.5 -> 2, 2.5 -> 2)
export function pyRound(value) {
  const floor = Math.floor(value);
  const diff = value - floor;
  if (diff > 0.5) return floor + 1;
  if (diff < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}
