import React, { useRef, useEffect, useState } from 'react';
import Webcam from 'react-webcam';
import { FaceLandmarker, HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import useSound from 'use-sound'; 
import alertSound from './alert.mp3';

const FacePickingDetector = () => {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [alert, setAlert] = useState(false);
  const [loading, setLoading] = useState(true);
  const faceLandmarkerRef = useRef(null);
  const handLandmarkerRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);
  const animationRef = useRef(null);
  const isMounted = useRef(true);


  const FACE_REGIONS = {
    CHEEK_INNER: [116, 123, 147, 352, 376, 433],
    NOSE: [4, 6, 168, 49, 64, 97, 2],
    MOUTH: [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
    CHIN_CENTER: [200, 429, 436, 365, 397],
    FOREHEAD_CENTER: [10, 151, 338],
    NECK: [152, 148, 176, 150, 149, 176, 377, 400, 378, 379, 365, 397, 288, 435, 367],
    EAR_LEFT: [234, 227, 93, 132, 58],
    EAR_RIGHT: [454, 447, 356, 323, 288]
  };

  const REGION_THRESHOLDS = {
    CHEEK_INNER: 0.08,
    NOSE: 0.07,
    MOUTH: 0.07,
    CHIN_CENTER: 0.07,
    FOREHEAD_CENTER: 0.1,
    NECK: 0.08,
    EAR_LEFT: 0.07,
    EAR_RIGHT: 0.07
  };

  const [playAlert, { stop }] = useSound(alertSound, { 
    volume: 0.7,
    interrupt: false
  });
  const lastPlayedRef = useRef(0);

  
  useEffect(() => {
    const initializeModels = async () => {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
      );
      
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
    if (!isMounted.current) return;

    if (!webcamRef.current || !canvasRef.current) return;
  
    if (!faceLandmarkerRef.current || !handLandmarkerRef.current) {
      console.log('Models not initialized yet');
      return;
    }
    const video = webcamRef.current.video;
    if (video.currentTime === lastVideoTimeRef.current) {
      animationRef.current = requestAnimationFrame(detect);
      return;
    }
    
    if (video.readyState < 2) {
      animationRef.current = requestAnimationFrame(detect);
      return;
    }
    
    lastVideoTimeRef.current = video.currentTime;
    
    const faceResults = faceLandmarkerRef.current.detectForVideo(video, Date.now());
    const handResults = handLandmarkerRef.current.detectForVideo(video, Date.now());

    drawLandmarks(faceResults, handResults);
    checkFacePicking(faceResults, handResults);

    animationRef.current = requestAnimationFrame(detect);
  };

  const drawLandmarks = (faceResults, handResults) => {
    const ctx = canvasRef.current.getContext('2d');
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    
    ctx.save();
    ctx.scale(-1, 1); 
    ctx.translate(-canvasRef.current.width, 0);
    
    // if (faceResults.faceLandmarks) {
    //   faceResults.faceLandmarks.forEach(landmarks => {
    //     landmarks.forEach((landmark, idx) => {
    //       if (idx % 10 === 0) {
    //         drawPoint(ctx, landmark.x * canvasRef.current.width, 
    //                   landmark.y * canvasRef.current.height, 2, 'green');
    //       }
    //     });
    //   });
    // }

    if (handResults.landmarks) {
      handResults.landmarks.forEach(landmarks => {
        landmarks.forEach(landmark => {
          drawPoint(ctx, landmark.x * canvasRef.current.width, 
                    landmark.y * canvasRef.current.height, 3, 'red');
        });
      });
    }

    ctx.restore(); 
  };

  const checkFacePicking = (faceResults, handResults) => {
    let isPicking = false;
    
    faceResults.faceLandmarks?.forEach(face => {
      handResults.landmarks?.forEach(hand => {
        const fingerTip = hand[8];
        Object.entries(FACE_REGIONS).forEach(([regionName, landmarks]) => {
          landmarks.forEach(faceIdx => {
            const faceLandmark = face[faceIdx];
            const distance = calculateDistance(
              fingerTip.x, fingerTip.y,
              faceLandmark.x, faceLandmark.y
            );
            
            if (distance < REGION_THRESHOLDS[regionName]) {
              isPicking = true;
            }
          });
        });
      });
    });
  
    if (isPicking && !alert) {
      const now = Date.now();
      if (now - lastPlayedRef.current > 1000) {
        playAlert();
        lastPlayedRef.current = now;
      }
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
