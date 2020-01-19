let y2 = 0;
let gaussian_previous = false;
tf.setBackend("cpu");

randomBetween = function(min, max) {
  let rand;
  rand = Math.random();
  if (typeof min === 'undefined') {
    return rand;
  } else if (typeof max === 'undefined') {
    if (min instanceof Array) {
      return min[Math.floor(rand * min.length)];
    } else {
      return rand * min;
    }
  } else {
    if (min > max) {
      const tmp = min;
      min = max;
      max = tmp;
    }
    return rand * (max - min) + min;
  }
};

randomGauss = function(mean, sd) {
  let y1, x1, x2, w;
  if (gaussian_previous) {
    y1 = y2;
    gaussian_previous = false;
  } else {
    do {
      x1 = this.randomBetween(2) - 1;
      x2 = this.randomBetween(2) - 1;
      w = x1 * x1 + x2 * x2;
    } while (w >= 1);
    w = Math.sqrt(-2 * Math.log(w) / w);
    y1 = x1 * w;
    y2 = x2 * w;
    gaussian_previous = true;
  }

  const m = mean || 0;
  const s = sd || 1;
  return y1 * s + m;
};

class NeuralNetwork {
  constructor(a, b, c, d) {
    if (a instanceof tf.Sequential) {
      this.model = a;
      this.input_nodes = b;
      this.hidden_nodes = c;
      this.output_nodes = d;
    } else {
      this.input_nodes = a;
      this.hidden_nodes = b;
      this.output_nodes = c;
      this.model = this.createModel();
    }
  }

  copy() {
    return tf.tidy(() => {
      const modelCopy = this.createModel();
      const weights = this.model.getWeights();
      const weightCopies = [];
      for (let i = 0; i < weights.length; i++) {
        weightCopies[i] = weights[i].clone();
      }
      modelCopy.setWeights(weightCopies);
      return new NeuralNetwork(
        modelCopy,
        this.input_nodes,
        this.hidden_nodes,
        this.output_nodes
      );
    });
  }

  mutate(rate) {
    tf.tidy(() => {
      const weights = this.model.getWeights();
      const mutatedWeights = [];
      for (let i = 0; i < weights.length; i++) {
        let tensor = weights[i];
        let shape = weights[i].shape;
        let values = tensor.dataSync().slice();
        for (let j = 0; j < values.length; j++) {
          if (randomBetween(1) < rate) {
            let w = values[j];
            values[j] = w + randomGauss();
          }
        }
        let newTensor = tf.tensor(values, shape);
        mutatedWeights[i] = newTensor;
      }
      this.model.setWeights(mutatedWeights);
    });
  }

  dispose() {
    this.model.dispose();
  }

  predict(inputs) {
    return tf.tidy(() => {
      const xs = tf.tensor2d([inputs]);
      const ys = this.model.predict(xs);
      const outputs = ys.dataSync();
      // console.log(outputs);
      return outputs;
    });
  }

  saveModelToLocalhost() {
    this.model.save('localstorage://population-best')
  }

  saveModelToFile() {
    this.model.save('downloads://population-best')
  }

  createModel() {
    const model = tf.sequential();
    const hidden = tf.layers.dense({
      units: this.hidden_nodes,
      inputShape: [this.input_nodes],
      activation: 'sigmoid'
    });
    model.add(hidden);
    const output = tf.layers.dense({
      units: this.output_nodes,
      activation: 'softmax'
    });
    model.add(output);
    return model;
  }
}

class Member {
  constructor(brain, mutateRate, a, b, c) {
    this.score = 0;
    this.fitness = 0;
    this.mutateRate = mutateRate;
    if (brain) {
      this.brain = brain.copy();
    } else {
      this.brain = new NeuralNetwork(a, b, c);
    }
  }

  dispose() {
    this.brain.dispose();
  }

  mutate() {
    this.brain.mutate(this.mutateRate);
  }

  think(inputs) {
    let output = this.brain.predict(inputs);
    return output;
  }

  saveModelToLocalhost() {
    this.brain.saveModelToLocalhost()
  }

  saveModelToFile() {
    this.brain.saveModelToFile()
  }
}

class Population {
  constructor(number, config) {
    this.TOTAL = number;
    this.memberConfig = config;
    this.generation = [];
    this.savedGeneration = [];
    this.isGenerationEmpty = false;
    this.initGeneration();
  }

  initGeneration() {
    for (let i = 0; i < this.TOTAL; i++) {
      this.generation[i] = new Member(null, this.memberConfig.mutateRate, this.memberConfig.inputNodes, this.memberConfig.hiddenNodes, this.memberConfig.outputNodes);
    }
  }

  removeMember(index) {
    this.savedGeneration.push(this.generation.splice(index, 1)[0]);
    if(!this.generation || this.generation.length === 0) this.isGenerationEmpty = true
  }

  getMemberPredict(index, inputs) {
    return this.generation[index].think(inputs);
  }

  scoreMember(index, addedScore) {
    this.generation[index].score += addedScore;
  }

  nextGeneration() {
    console.log('next generation');
    this.calculateFitness();
    for (let i = 0; i < TOTAL; i++) {
      this.generation[i] = this.pickOne();
    }
    for (let i = 0; i < TOTAL; i++) {
      this.savedGeneration[i].dispose();
    }
    this.savedGeneration = [];
    this.isGenerationEmpty = false;
  }
  
  pickOne() {
    let index = 0;
    let r = random(1);
    while (r > 0) {
      r = r - this.savedGeneration[index].fitness;
      index++;
    }
    index--;
    let member = this.savedGeneration[index];
    let child = new Member(member.brain, this.memberConfig.mutateRate, this.memberConfig.inputNodes, this.memberConfig.hiddenNodes, this.memberConfig.outputNodes);
    child.mutate();
    return child;
  }
  
  calculateFitness() {
    let sum = 0;
    for (let member of this.savedGeneration) {
      sum += member.score;
    }
    for (let member of this.savedGeneration) {
      member.fitness = member.score / sum;
    }
  }

  saveBestModel() {
    this.generation[0].saveModelToLocalhost()
    this.generation[0].saveModelToFile()
  }
}
