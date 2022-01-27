import { StatusBar } from 'expo-status-bar';
import React, { useState, useEffect }  from 'react';
import { StyleSheet, Text, View, Image, Button } from 'react-native';
//import Svg, {Rect} from 'react-native-svg';
import * as tf from '@tensorflow/tfjs'
import { fetch, bundleResourceIO } from '@tensorflow/tfjs-react-native';
//import { bundleResourceIO } from '@tensorflow/tfjs-react-native';
import * as blazeface from '@tensorflow-models/blazeface';
import jpeg from 'jpeg-js'

// import * as mobilenet from '@tensorflow-models/mobilenet'
// import {TFLiteImageRecognition} from 'react-native-tensorflow-lite';
// import {TFLiteImageRecognition} from 'tflite-react-native';

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'dodgerblue',
    alignItems: 'center',
    justifyContent: 'center',
  },

});


export default function App() {
  console.log("App Started")

  const emotions =  ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'] // ['Happy', 'Disgust', 'Fear', 'Sad', 'Angry', 'Surprise', 'Neutral']


  const [faces,setFaces]=useState([])
  const [faceDetector,setFaceDetector]=useState("")
  const [emotionDetector,setEmotionDetector]=useState("")
  
  useEffect(() => {
    async function loadModel(){
      console.log("[+] Application started")
      
      //Wait for tensorflow module to be ready
      const tfReady = await tf.ready();
      console.log("[+] Loading custom emotion detection model")
      
      //Replce model.json and group1-shard.bin with your own custom model
      const modelJson =  ("http://192.168.29.69:8000/model/model2.json");//require("./assets/model/model2.json");
      console.log(modelJson);
    // const modelWeight =  require("./assets/model/group2.bin");
     const emotionDetector = await tf.loadLayersModel(modelJson);
     emotionDetector.summary()
      console.log("[+] Loading pre-trained face detection model : ")
      
      //Blazeface is a face detection model provided by Google
      const faceDetector =  await blazeface.load();
      console.log("facedetector loaded")
  //     //Assign model to variable
     setEmotionDetector(emotionDetector)
      console.log("Emotion detector set")
      setFaceDetector(faceDetector)
      console.log("[+] Model Loaded")
    }
    loadModel()
  }, []); 


  function imageToTensor(rawImageData){
  //   //Function to convert jpeg image to tensors
    const TO_UINT8ARRAY = true;
    const { width, height, data } = jpeg.decode(rawImageData, TO_UINT8ARRAY);
    
  //   // Drop the alpha channel info for mobilenet
    const buffer = new Uint8Array(width * height * 3);
    let offset = 0; // offset into original data
    for (let i = 0; i < buffer.length; i += 3) {
      buffer[i] = data[offset];
      buffer[i + 1] = data[offset + 1];
      buffer[i + 2] = data[offset + 2];
      offset += 4;
    }
    return tf.tensor3d(buffer, [height, width, 3]);
  }

  function imageToGrayscaleTensor(rawImageData){
    //   //Function to convert jpeg image to tensors
      const TO_UINT8ARRAY = true;
      const { width, height, data } = jpeg.decode(rawImageData, TO_UINT8ARRAY);
      
    //   // Drop the alpha channel info for mobilenet
      const buffer = new Uint8Array(width * height * 1);
      let offset = 0; // offset into original data
      for (let i = 0; i < buffer.length; i++) {
        buffer[i] = 0.2989*data[offset] + 0.5870*data[offset + 1] + 0.1140*data[offset + 2];
        offset += 4;
      }
      return tf.tensor3d(buffer, [height, width, 1]);
    }
     




  const getFaces = async() => {
//try{
     // const imageLink = 'https://upload.wikimedia.org/wikipedia/commons/thumb/7/7d/Tigray_Woman_%288756820602%29.jpg/800px-Tigray_Woman_%288756820602%29.jpg'
  //const imageLink = 'https://thumbs.dreamstime.com/z/two-friends-standing-together-over-pink-background-happy-face-smiling-crossed-arms-looking-camera-positive-person-214021923.jpg'
      
      const imageLink = 'http://192.168.29.69:8000/images/angry.jpg';
      console.log("[+] Retrieving image from link :"+imageLink)
      
      const response = await fetch(imageLink, {}, { isBinary: true });
     
      console.log("response : ",response);

      const rawImageData = await response.arrayBuffer();

      console.log("here");
      let imageTensor = imageToTensor(rawImageData).resizeBilinear([224,224]);
      console.log("image tensor");

      const faces = await faceDetector.estimateFaces(imageTensor, false);

    

      console.log("number of faces : ",faces.length);

      let tempArray = []
      //Loop through the available faces, check if the person is wearing a mask. 
      for (let i=0;i<faces.length;i++){
        let color = "red"
        let width = parseInt((faces[i].bottomRight[1] - faces[i].topLeft[1]))
        let height = parseInt((faces[i].bottomRight[0] - faces[i].topLeft[0]))
        let faceTensor=imageTensor.slice([parseInt(faces[i].topLeft[1]),parseInt(faces[i].topLeft[0]),0],[width,height,3])
        // tf.browser.toPixels(faceTensor).then(r=>{
        //     console.log("Response : ",r);
        // });
        
        //faceTensor = faceTensor.resizeBilinear([224,224]).reshape([1,224,224,3])
        
        console.log("faceTensor", (faceTensor));
        
        let grayscale_tensor =  faceTensor.mean(2,true);
        console.log("grayscale_tensor", JSON.stringify(grayscale_tensor));
        grayscale_tensor = grayscale_tensor.resizeBilinear([48,48]).reshape([1,48,48,1])
        //grayscale_tensor = grayscale_tensor.reshape([1,48,48,1])
        let result = await emotionDetector.predict(grayscale_tensor).data()

        console.log("Prediction result : ",JSON.stringify(result))

        //if result[0]>result[1], the emotion is detected 
        
        let key=-1,max = -1;
        result.forEach( (element,idx) => {
            if(element>max)
            {
              max = element;
              key = idx;
            }
        });

        console.log("Predicted Emotion : ",emotions[key]);
        
        tempArray.push({
          id:i,
          location:faces[i],
          color:color
        })
      }
      setFaces(tempArray)
      console.log("[+] Prediction Completed")
    // }catch{
    //   console.log("[-] Unable to load image")
    // }
    
  }

  return (
    <View style={styles.container}>

      <Text>Facial Emotion</Text>
      <View style = {styles.image}>
        <Image style={{width: 200, height: 200}} source={require('./assets/images/happy.jpg')}/>
      </View>
      <StatusBar style="auto" />

      
      <Button 
        title="Prediction"
        onPress={()=>{getFaces()}}
        // disabled={!isEnabled}
      />
      
    </View>
  );


  
  // // myImgClassifier = MyImageClassifier.create();
  
  // // myImgClassifier.classifyImage();
  // // res = myImgClassifier.state.name
  // // DisplayAnImage

  // const [displayText, setDisplayText] = useState('Loading')

  // async function getPrediction(){
  //   setDisplayText("Loading Tensorflow")
  //   await tf.ready()
  //   setDisplayText("Loading Mobile Net")

  //   const model = await mobilenet.load()
    

  // }
}


























// class DisplayAnImage extends Component {
//   render() {
//     return (
//       <View style={styles.container}>
//         <Image
//           source={require('@expo/images/icon.png')}
//         />
//       </View>
//     );
//   }
// }


// class MyImageClassifier extends Component {
 
//   constructor() {
//     super()
//     this.state = {}
 
//     try {
//       // Initialize Tensorflow Lite Image Recognizer
//       this.classifier = new TFLiteImageRecognition({
//         model: "model_lite.tflite",  // Your tflite model in assets folder.
//         labels: "labels.txt" // Your label file
//       })
 
//     } catch(err) {
//       alert(err)
//     }
//   }
//   componentWillMount() {
//     this.classifyImage("images/happy.jpg") // Your image path.
//   }
  
//   // async classifyImage(imagePath) {
//   classifyImage(imagePath) {
//     try {
//       // const results = await this.classifier.recognize({
//       const results = this.classifier.recognize({
//         image: imagePath, // Your image path.
//         inputShape: 48, // the input shape of your model. If none given, it will be default to 224.
//       })
 
//       const resultObj = {
//                 name: "Name: " + results[0].name,  
//                 confidence: "Confidence: " + results[0].confidence, 
//                 inference: "Inference: " + results[0].inference + "ms"
//             };
//       this.setState(resultObj)
      
//     } catch(err) {
//       alert(err)
//     }   
//   }
  
//   componentWillUnmount() {
//     this.classifier.close() // Must close the classifier when destroying or unmounting component to release object.
//   }
 
//   render() {
//     return (
//       <View style={styles.container}>
//         <View>
//           <Text style={styles.results}>
//             {this.state.name}
//           </Text>
//           <Text style={styles.results}>
//             {this.state.confidence}
//           </Text>
//           <Text style={styles.results}>
//             {this.state.inference}
//           </Text>
//         </View>
//       </View>
//     );
//   }
// }