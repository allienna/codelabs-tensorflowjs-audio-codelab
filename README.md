# [Codelab] Audio recognition using transfer learning

Information can be found [here](https://codelabs.developers.google.com/codelabs/tensorflowjs-audio-codelab/index.html?index=..%2F..index#0).

While Google could remove instructions, I copy/paste them below.

## About this codelab
Written by Daniel Smilkov, Nikhil Thorat, Ann Yuan

## 1. Introduction

In this codelab, you will build an audio recognition network and use it to control a slider in the browser by making sounds.
You will be using Tensorflow.js, a powerful and flexible machine learning library for Javascript.

First, you will load and run a [pre-trained model](https://github.com/tensorflow/tfjs-models/tree/master/speech-commands
that can recognize 20 speech commands. Then using your microphone, you will build and train a simple neural network that recognizes
your sounds and make the slider go left or right.

This codelab will **not** go over the theory behind audio recognition models. If you are curious about that, 
check out [this tutotial](https://www.tensorflow.org/tutorials/audio/simple_audio).

We have also created a [glossary](https://docs.google.com/document/d/1YcLoCbDFfE6qJGZikBbjfbiI_dG4iETyIBEcS-NQbNE/edit) 
of machine learning terms that you find in this codelab.

### What you'll learn
- How to load a pre-trained speech command recognition model
- How to make real-time predictions using the microphone
- How to train and use a custom audio recognition model using the browser microphone

So let's get started

## 2. Requirements

To complete this codelab, you will need:
1. A recent version of Chrome or another modern browser.
2. A text editor, either running locally on your machine or on the web via something like Codepen or Glitch.
3. Knowledge of HTML, CSS, JavaScript, and Chrome DevTools (or your preferred browsers devtools)
4. A high-level conceptual understanding of Neural Networks. If you need an intrnoduction or refresher, consider watching
[this video by 3blue1brown](https://www.youtube.com/watch?v=aircAruvnKk) or this
[video on Deep Learning in Javascript by Ashi Krishna](https://www.youtube.com/watch?v=SV-cgdobtTA).

> **Note**: If you are at a CodeLab kiosk we recommend using [glitch.com](glitch.com) to complete this codelab
> We have set up a [starter project for you to remix](https://glitch.com/~tensorflow-js-boilerplate) that loads tensorflow.js.

## 3. Load TensorFlow.js and the Audio model

Open `index.html` in an editor and add this content:
```html
<html>
  <head>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/speech-commands"></script>
  </head>
  <body>
    <div id="console"></div>
    <script src="index.js"></script>
  </body>
</html>
```
The first `<script>` tag imports the TensorFlow.js library, and the second `<script>` imports the pre-trained 
[Speech Commands model](https://github.com/tensorflow/tfjs-models/tree/master/speech-commands). 
The `<div id="console">` tag will be used to display the output of the model.

## 4. Predict in real-time

Next, open/create the file `index.js` in a code editor, and include the following code:
```javascript
let recognizer;

function predictWord() {
    // Array of words that the recognizer is trained to recognize.
    const words = recognizer.wordLabels();
    recognizer.listen(({scores}) => {
        // Turns scores into a list of (score,word) pairs.
        scores = Array.from(scores).map((s, i) => ({score: s, word: words[i]}))
        // Find the most probable word.
        scores.sort((s1, s2) => s2.score - s1.score);
        document.querySelector('#console').textContent = scores[0].word;
    }, {probabilityThreshold: 0.75});
}

async function app() {
    recognizer = speechCommands.create('BROWSER_FFT');
    await recognizer.ensureModelLoaded();
    predictWord();
}

app();
```

## 5. Test the prediction

Make sure your device has a microphone. It's worth noting that this will work on a mobile phone! To run the webpage, 
open **index.html** in a browser. If you are working from a local file, to access the microphone you will have to start
a web server and use `http://localhost:port/`.

To start a simple webserver on port 8000:

`python -m SimpleHTTPServer`

> **Note**: with Python 3, you should use `python -m http.server --bind 127.0.0.1`

It may take a bit of time to download the model, so please be patient. As soon as the model loads, you should see a 
word on the top of the page. The model was trained to recognize the numbers 0 through 9 and a few additional commands
such as "left", "right", "yes", "no", etc.

Speak one of those words. Does it get your word correctly? Play with the `probalityThreshold` which controls how often
the model fires - 0.75 means that the model will fire when it is more than 75% confident that it hears a given word.

To learn more about the Speech Commands model and its API, see the 
[README.md](https://github.com/tensorflow/tfjs-models/blob/master/speech-commands/README.md) on Github.

## 6. Collect data

To make it fun, let's use short sounds instead of whole words to control the slides!

You are going to train a model to recognize 3 different command: "Left", "Right", "Noise" which make the slide move left
or right. Recognizing "Noise" (no action needed) is critical in speech detection since we want the slider to react only 
when we produce the right sound, and not when we are generally speaking and moving around.

1. First we need to collect data. Add a simple UI to the app by adding this inside the `<body>` tag before the `<div id="console">`:
```html
<button id="left" onmousedown="collect(0)" onmouseup="collect(null)">Left</button>
<button id="right" onmousedown="collect(1)" onmouseup="collect(null)">Left</button>
<button id="noise" onmousedown="collect(2)" onmouseup="collect(null)">Left</button>
```
2. Add this to `index.js`:
```javascript
// One frame is ~23ms of audio.
const NUM_FRAMES = 3;
let examples = [];

function collect(label) {
 if (recognizer.isListening()) {
   return recognizer.stopListening();
 }
 if (label == null) {
   return;
 }
 recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
   let vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
   examples.push({vals, label});
   document.querySelector('#console').textContent =
       `${examples.length} examples collected`;
 }, {
   overlapFactor: 0.999,
   includeSpectrogram: true,
   invokeCallbackOnNoiseAndUnknown: true
 });
}

function normalize(x) {
 const mean = -100;
 const std = 10;
 return x.map(x => (x - mean) / std);
}
```
3. Remove `predictWord()` from `app()`:
```javascript
async function app() {
 recognizer = speechCommands.create('BROWSER_FFT');
 await recognizer.ensureModelLoaded();
 // predictWord() no longer called.
}
```

### Breaking it down

This code might be overwhelming at first, so let's break it down.

We've added three buttons to our UI labeled "Left", "Right", "Noise", corresponding to the three commands we want
our model to recognize. Pressing these buttons call our newly added `collect()` function, which creates training examples
for our model.

`collect()` associates a `label` with the output of `recognizer.listen()`. Since `includeSpectrogram` is true, 
`recognizer.listen()` gives the raw spectrogram (frequency data) for 1 sec of audio, divided into 43 frames, so each
frame is ~23ms of audio:
```javascript
recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
...
}, {includeSpectrogram: true});
```

Since we want to use short sounds instead of words to control the slider, we are taking into consideration only the latest
3 frames (~70ms):
```javascript
let vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
```

And to avoid numerical issues, we normalize the data to have an average of 0 and a standard deviation of 1. In this case,
the spectrogram values are usually large negative numbers around -100 and deviation of 10:
```javascript
const mean = -100;
const std = 10;
return x.map(x => (x - mean) / std);
```

Finally, each training example will have 2 fields:
- `label`****: 0, 1 and 2 for "Left", "Right" and "Noise" respectively.
- `vals`****: 696 numbers holding the frequency information (spectrogram)

and we store all data in the `examples` variable:
```javascript
examples.push({vals, label});
``` 

## 7. Test data collection

> **Note**: After testing data collection you will throw your samples away so don't waste time collecting too many! 

Open **index.html** in a browser, and you should see 3 buttons corresponding to the 3 commands. 
If you are working from a local file, to access the microphone you will have to start a webserver and use `http://localhost:port/`.

To start a simple webserver on port 8000:

`python -m SimpleHTTPServer`

> **Note**: with Python 3, you should use `python -m http.server --bind 127.0.0.1`

To collect examples for each command, make a consistent sound repeatedly (or continuously) while **pressing and holding** 
each button for 3-4 seconds. You should collect ~150 examples for each label. For example, we can snap fingers for "Left",
whistle for "Right", and alternate between silence and talk for "Noise".

As you collect more examples, for counter shown on the page should go up. Feel free to also inspect the data by calling 
console.log() on the `examples` variable in the console. At this point the goal is to test the data collection process.
Later you will re-collect data when you are testing the whole app.

## 8. Train a model

1. Add a "**Train**" button right after the "**Noise**" button in the body in **index.html**:
```javascript
<br/><br/>
<button id="train" onclick="train()">Train</button>
```
2. Add the following to the existing code in **index.js**:
```javascript
const INPUT_SHAPE = [NUM_FRAMES, 232, 1];
let model;

async function train() {
 toggleButtons(false);
 const ys = tf.oneHot(examples.map(e => e.label), 3);
 const xsShape = [examples.length, ...INPUT_SHAPE];
 const xs = tf.tensor(flatten(examples.map(e => e.vals)), xsShape);

 await model.fit(xs, ys, {
   batchSize: 16,
   epochs: 10,
   callbacks: {
     onEpochEnd: (epoch, logs) => {
       document.querySelector('#console').textContent =
           `Accuracy: ${(logs.acc * 100).toFixed(1)}% Epoch: ${epoch + 1}`;
     }
   }
 });
 tf.dispose([xs, ys]);
 toggleButtons(true);
}

function buildModel() {
 model = tf.sequential();
 model.add(tf.layers.depthwiseConv2d({
   depthMultiplier: 8,
   kernelSize: [NUM_FRAMES, 3],
   activation: 'relu',
   inputShape: INPUT_SHAPE
 }));
 model.add(tf.layers.maxPooling2d({poolSize: [1, 2], strides: [2, 2]}));
 model.add(tf.layers.flatten());
 model.add(tf.layers.dense({units: 3, activation: 'softmax'}));
 const optimizer = tf.train.adam(0.01);
 model.compile({
   optimizer,
   loss: 'categoricalCrossentropy',
   metrics: ['accuracy']
 });
}

function toggleButtons(enable) {
 document.querySelectorAll('button').forEach(b => b.disabled = !enable);
}

function flatten(tensors) {
 const size = tensors[0].length;
 const result = new Float32Array(tensors.length * size);
 tensors.forEach((arr, i) => result.set(arr, i * size));
 return result;
}
```
3. Call `buildModel()` when the app loads:
```javascript
async function app() {
 recognizer = speechCommands.create('BROWSER_FFT');
 await recognizer.ensureModelLoaded();
 // Add this line.
 buildModel();
}
```

At this point if you refresh the app you'll see a new "**Train**" button. You can test training by re-collecting data 
and clicking "Train", or you can wait until step 10 to test training along with prediction.

### Breaking it down
At a high level we are doing two things: `buildModel()` defines the model architecture and `train()` trains the model 
using the collected data.

#### Model architecture
The model has 4 layers: a convolutional layer that processes the audio data (represented as a spectrogram), a max pool 
layer, a flatten layer, and a dense layer that maps to the 3 actions:
```javascript
model = tf.sequential();
 model.add(tf.layers.depthwiseConv2d({
   depthMultiplier: 8,
   kernelSize: [NUM_FRAMES, 3],
   activation: 'relu',
   inputShape: INPUT_SHAPE
 }));
 model.add(tf.layers.maxPooling2d({poolSize: [1, 2], strides: [2, 2]}));
 model.add(tf.layers.flatten());
 model.add(tf.layers.dense({units: 3, activation: 'softmax'}));
```

The input shape of the model is `[NUM_FRAMES, 232, 1]` where each frame is 23ms of audio containing 232 numbers that
correspond to different frequencies (232 was chosen because it is the amount of frequency buckets needed to capture 
the human voice). In this codelab, we are using samples that are 3 frames long (~70ms samples) since we are making 
sounds instead of speaking whole words to control the slider.

We compile our model to get it ready for training:

```javascript
const optimizer = tf.train.adam(0.01);
 model.compile({
   optimizer,
   loss: 'categoricalCrossentropy',
   metrics: ['accuracy']
 });
```
We use the [Adam optimizer](https://arxiv.org/abs/1412.6980), a common optimizer used in deep learning, and 
`categoricalCrossEntropy` for loss, the standard loss function used for classification. In short, it measures how far
the predicted probabilities (one probability per class) are from having 100% probability in the true class, and 0% 
probability for all the other classes. We also provide `accuracy` as a metric to monitor, which will give us the percentage 
of examples the model gets correct after each epoch of training.

#### Training
The training goes 10 times (epochs) over the data using a batch size of 16 (processing 16 examples at a time) and 
shows the current accuracy in the UI:
```javascript
await model.fit(xs, ys, {
   batchSize: 16,
   epochs: 10,
   callbacks: {
     onEpochEnd: (epoch, logs) => {
       document.querySelector('#console').textContent =
           `Accuracy: ${(logs.acc * 100).toFixed(1)}% Epoch: ${epoch + 1}`;
     }
   }
 });
```

## 9. Update the slider in real-time

Now that we can train our model, let's add code to make predictions in real-time and move the slider. 
Add this right after the "**Train**" button in **index.html**:

```html
<br/><br/>
<button id="listen" onclick="listen()">Listen</button>
<input type="range" id="output" min="0" max="10" step="0.1">
```
And the following in index.js:
```javascript
async function moveSlider(labelTensor) {
 const label = (await labelTensor.data())[0];
 document.getElementById('console').textContent = label;
 if (label == 2) {
   return;
 }
 let delta = 0.1;
 const prevValue = +document.getElementById('output').value;
 document.getElementById('output').value =
     prevValue + (label === 0 ? -delta : delta);
}

function listen() {
 if (recognizer.isListening()) {
   recognizer.stopListening();
   toggleButtons(true);
   document.getElementById('listen').textContent = 'Listen';
   return;
 }
 toggleButtons(false);
 document.getElementById('listen').textContent = 'Stop';
 document.getElementById('listen').disabled = false;

 recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
   const vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
   const input = tf.tensor(vals, [1, ...INPUT_SHAPE]);
   const probs = model.predict(input);
   const predLabel = probs.argMax(1);
   await moveSlider(predLabel);
   tf.dispose([input, probs, predLabel]);
 }, {
   overlapFactor: 0.999,
   includeSpectrogram: true,
   invokeCallbackOnNoiseAndUnknown: true
 });
}
```

### Breaking it down
#### Real-time prediction
`listen()` listens to the microphone and makes real time predictions. 
The code is very similar to the `collect()` method, which normalizes the raw spectrogram and drops all but the last 
`NUM_FRAMES` frames. The only difference is that we also call the trained model to get a prediction:
```javascript
const probs = model.predict(input);
const predLabel = probs.argMax(1);
await moveSlider(predLabel);
```
The output of `model.predict(input)`is a Tensor of shape `[1, numClasses]` representing a probability distribution over 
the number of classes. More simply, this is just a set of confidences for each of the possible output classes which sum to 1. 
The Tensor has an outer dimension of 1 because that is the size of the batch (a single example).

To convert the probability distribution to a single integer representing the most likely class, 
we call `probs.argMax(1)`which returns the class index with the highest probability. We pass a "1" as the axis parameter
because we want to compute the `argMax` over the last dimension, `numClasses`.

#### Updating the slider

`moveSlider()` decreases the value of the slider if the label is 0 ("Left") , increases it if the label is 1 ("Right")
and ignores if the label is 2 ("Noise").

#### Disposing tensors

To clean up GPU memory it's important for us to manually call tf.dispose() on output Tensors. The alternative to manual
`tf.dispose()` is wrapping function calls in a `tf.tidy()`, but this cannot be used with async functions.
```javascript
   tf.dispose([input, probs, predLabel]);
```

## 10. Test the final app

Open **index.html** in your browser and collect data as you did in the previous section with the 3 buttons corresponding 
to the 3 commands. Remember to **press and hold** each button for 3-4 seconds while collecting data.

Once you've collected examples, press the "**Train**" button. This will start training the model and you should see the 
accuracy of the model go above 90%. If you don't achieve good model performance, try collecting more data.

Once the training is done, press the "**Listen**" button to make predictions from the microphone and control the slider!

See more tutorials at [http://js.tensorflow.org/](http://js.tensorflow.org/).