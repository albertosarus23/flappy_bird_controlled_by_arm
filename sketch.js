let video;
let poses = [];
let brain;
let skeleton;
let pose;

let state = 'waiting';
let targetLabel;
let poseNet;

let poseLabel = "ing";
/************************** training section *************************

function gotPoses(poses) {
  // console.log(poses); 
  if (poses.length > 0) {
    pose = poses[0].pose;
    //skeleton = poses[0].skeleton;
    if (state == 'collecting') {
      let inputs = [];
      for (let i = 0; i < pose.keypoints.length; i++) {
        let x = pose.keypoints[i].position.x;
        let y = pose.keypoints[i].position.y;
        inputs.push(x);
        inputs.push(y);
      }
      let target = [targetLabel];
      brain.addData(inputs, target);
    }
  }
}

function keyPressed(){
  if (key == 's'){
    brain.saveData();
    console.log('data saved');
  }
  else{
    targetLabel = key;
    console.log(targetLabel);
    setTimeout(function(){
      console.log('collecting');
      state = 'collecting';
      setTimeout(function(){
        console.log('collected');
        state = 'waiting';
      }, 10000);
    }, 10000);
  }
}
************************************************/

function setup() {
  createCanvas(640, 480);
  video = createCapture(VIDEO);
  video.size(width, height);
  poseNet = ml5.poseNet(video, modelLoaded);
  //const poseNet = ml5.poseNet(video, { flipHorizontal: true });

  poseNet.on('pose', gotPoses);

  // Hide the video element, and just show the canvas
  video.hide();
  let options = {
    inputs: 34,
    outputs: 3,
    task: 'classification',
    debug: true
  }
  brain = ml5.neuralNetwork(options);
  // LOAD TRAINING DATA
  //brain.loadData('training_set_final.json',dataReady);

  // LOAD PRETRAINED MODEL
  const modelInfo = {
    model: 'model/model.json',
    metadata: 'model/model_meta.json',
    weights: 'model/model.weights.bin',
  };
  brain.load(modelInfo, brainLoaded);
}

function classifyPose() {
  if (pose) {
    let inputs = [];
    for (let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      inputs.push(x);
      inputs.push(y);
    }
    brain.classify(inputs, gotResult);
  } else {
    setTimeout(classifyPose, 100);
  }
}

function gotResult(error, results) {
  if (results[0].confidence > 0.75) {
    poseLabel = results[0].label.toUpperCase();
  }

  //console.log(results[0].confidence);
  classifyPose();
}

function modelReady(){
  console.log('model ready!');
}

function brainLoaded() {
  console.log('pose classification ready!');
  classifyPose();
}


function dataReady(){
  brain.normalizeData();
  brain.train({epochs:50},finished);
}

function finished(){
  console.log('model trained');
  brain.save();
  classifyPose();
}

function gotPoses(poses) {
  if (poses.length > 0) {
    pose = poses[0].pose;
    skeleton = poses[0].skeleton;
  }
}

function modelLoaded() {
  console.log('poseNet ready');
}

function draw() {
  push();
  translate(video.width, 0);
  scale(-1, 1);
  image(video, 0, 0);
  pop();
  //keyPressed();
  //gotPoses(poses);
  fill(0, 0, 255);
  noStroke();
  textSize(512);
  textAlign(CENTER, CENTER);
  text(poseLabel, width / 2, height / 2);
  drawKeypoints(poses);
  drawSkeleton(poses);
}

////// drawing method //////////
function drawKeypoints(poses) {
  for (let pose of poses) {
    for (let keypoint of pose.pose.keypoints) {
      if (keypoint.score > 0.2) {
        fill(0, 255, 0);
        noStroke();
        ellipse(keypoint.position.x, keypoint.position.y, 30, 30);
      }
    }
  }
}
function drawSkeleton(poses) {
  for (let pose of poses) {
    for (let skeleton of pose.skeleton) {
      let p1 = skeleton[0];
			let p2 = skeleton[1];
      stroke(255, 0, 0);
      line(p1.position.x, p1.position.y, p2.position.x, p2.position.y);
    }
  }
}
