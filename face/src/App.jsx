import React, { useRef, useEffect, useState } from 'react';
import Webcam from 'react-webcam';
import { FaceLandmarker, HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import useSound from 'use-sound'; // New import
import alertSound from './alert.mp3'; // Import sound file

const FacePickingDetector = () => {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [alert, setAlert] = useState(false);
  const [loading, setLoading] = useState(true);
  const [playAlert] = useSound(alertSound, { volume: 0.7 }); // Sound hook
  const faceLandmarkerRef = useRef(null);
  const handLandmarkerRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);
  const animationRef = useRef(null);

  // Detection thresholds
  const PICK_DISTANCE_THRESHOLD = 0.1; // Normalized screen distance
  const FACE_REGIONS = {
    CHEEK: [123, 352], // Example face mesh indices
    NOSE: [4, 6],
    FOREHEAD: [10, 338]
  };

  useEffect(() => {
    const initializeModels = async () => {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
      );
      
      // Initialize Face Landmarker
      faceLandmarkerRef.current = await FaceLandmarker.createFromOptions(
        vision,
        {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
            delegate: "GPU"
          },
          outputFaceBlendshapes: true,
          runningMode: "VIDEO",
          numFaces: 1
        }
      );

      // Initialize Hand Landmarker
      handLandmarkerRef.current = await HandLandmarker.createFromOptions(
        vision,
        {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 2
        }
      );

      setLoading(false);
    };

    initializeModels().catch(console.error);

    return () => {
      cancelAnimationFrame(animationRef.current);
      if (faceLandmarkerRef.current) faceLandmarkerRef.current.close();
      if (handLandmarkerRef.current) handLandmarkerRef.current.close();
    };
  }, []);

  const detect = async () => {
    if (!webcamRef.current || !canvasRef.current) return;
    
    const video = webcamRef.current.video;
    if (video.currentTime === lastVideoTimeRef.current) {
      animationRef.current = requestAnimationFrame(detect);
      return;
    }
    
    lastVideoTimeRef.current = video.currentTime;
    
    // Detect face landmarks
    const faceResults = faceLandmarkerRef.current.detectForVideo(video, Date.now());
    // Detect hand landmarks
    const handResults = handLandmarkerRef.current.detectForVideo(video, Date.now());

    // Draw results and check proximity
    drawLandmarks(faceResults, handResults);
    checkFacePicking(faceResults, handResults);

    animationRef.current = requestAnimationFrame(detect);
  };

  const drawLandmarks = (faceResults, handResults) => {
    const ctx = canvasRef.current.getContext('2d');
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    
    // Draw face landmarks
    if (faceResults.faceLandmarks) {
      faceResults.faceLandmarks.forEach(landmarks => {
        landmarks.forEach((landmark, idx) => {
          if (idx % 10 === 0) { // Draw every 10th landmark
            drawPoint(ctx, landmark.x * canvasRef.current.width, 
                      landmark.y * canvasRef.current.height, 2, 'green');
          }
        });
      });
    }

    // Draw hand landmarks
    if (handResults.landmarks) {
      handResults.landmarks.forEach(landmarks => {
        landmarks.forEach(landmark => {
          drawPoint(ctx, landmark.x * canvasRef.current.width, 
                    landmark.y * canvasRef.current.height, 3, 'red');
        });
      });
    }
  };

  const checkFacePicking = (faceResults, handResults) => {
    if (!faceResults.faceLandmarks || !handResults.landmarks) return;

    let isPicking = false;
    
    faceResults.faceLandmarks.forEach(face => {
      handResults.landmarks.forEach(hand => {
        const fingerTip = hand[8];
        Object.values(FACE_REGIONS).forEach(region => {
          region.forEach(faceIdx => {
            const faceLandmark = face[faceIdx];
            const distance = calculateDistance(
              fingerTip.x, fingerTip.y,
              faceLandmark.x, faceLandmark.y
            );
            
            if (distance < PICK_DISTANCE_THRESHOLD) {
              isPicking = true;
            }
          });
        });
      });
    });

    if (isPicking && !alert) {
      playAlert(); // Trigger sound on new detection
    }
    setAlert(isPicking);
  };


  const calculateDistance = (x1, y1, x2, y2) => {
    return Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
  };

  const drawPoint = (ctx, x, y, radius, color) => {
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
  };

  return (
    <div className="detector-container">
      {loading && <div className="loading">Initializing models...</div>}
      
      <Webcam
        ref={webcamRef}
        mirrored
        onPlay={detect}
        style={{
          position: 'absolute',
          marginLeft: 'auto',
          marginRight: 'auto',
          left: 0,
          right: 0,
          textAlign: 'center',
          zIndex: 9,
          width: 640,
          height: 480
        }}
      />

      <canvas
        ref={canvasRef}
        style={{
          position: 'absolute',
          marginLeft: 'auto',
          marginRight: 'auto',
          left: 0,
          right: 0,
          textAlign: 'center',
          zIndex: 10,
          width: 640,
          height: 480
        }}
      />

      {alert && (
        <div className="alert-box">
          ⚠️ Face touching detected!
        </div>
      )}
    </div>
  );
};

export default FacePickingDetector;
