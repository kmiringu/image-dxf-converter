// @ts-nocheck
import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Upload, Settings, Download, Play, RotateCw, Zap, Target, Cpu, Eye, AlertTriangle, CheckCircle } from 'lucide-react';

const ProductionImageConverter = () => {
  const [image, setImage] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');
  const [results, setResults] = useState(null);
  const [rotation, setRotation] = useState(0);
  const [imageAnalysis, setImageAnalysis] = useState(null);
  const [errors, setErrors] = useState([]);
  const [config, setConfig] = useState({
    width: 300,
    height: 100,
    units: 'mm',
    precision: 'high',
    simplification: 'auto',
    processingPipeline: 'auto',
    backgroundThreshold: 75,
    edgeSensitivity: 70,
    contourMinLength: 20,
  });

  const canvasRefs = {
    original: useRef(null),
    processed: useRef(null),
    edges: useRef(null),
    preview: useRef(null),
    working: useRef(null)
  };

  const fileInputRef = useRef(null);

  // Production-grade image analysis
  const analyzeImageIntelligently = useCallback((imageData) => {
    const { data, width, height } = imageData;
    const totalPixels = width * height;
    
    let stats = {
      backgroundPixels: 0,
      edgePixels: 0,
      brightness: 0,
      contrast: 0
    };
    
    let brightnessValues = [];
    
    // Basic statistical analysis
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      
      const brightness = 0.299 * r + 0.587 * g + 0.114 * b;
      brightnessValues.push(brightness);
      
      if (r > 240 && g > 240 && b > 240) {
        stats.backgroundPixels++;
      }
    }
    
    // Calculate contrast
    const brightnessMean = brightnessValues.reduce((sum, val) => sum + val, 0) / brightnessValues.length;
    const variance = brightnessValues.reduce((sum, val) => sum + Math.pow(val - brightnessMean, 2), 0) / brightnessValues.length;
    stats.contrast = Math.sqrt(variance);
    stats.brightness = brightnessMean;
    
    // Edge detection analysis
    stats.edgePixels = detectEdgePixels(data, width, height);
    
    // Determine image type and optimal pipeline
    const backgroundRatio = stats.backgroundPixels / totalPixels;
    const edgeRatio = stats.edgePixels / totalPixels;
    const isHighContrast = stats.contrast > 50;
    const isCleanLineArt = backgroundRatio > 0.7 && edgeRatio > 0.05;
    const isPhotographic = backgroundRatio < 0.3 && stats.contrast < 30;
    const hasComplexBackground = backgroundRatio < 0.5 && stats.contrast > 40;
    
    let imageType, pipeline, confidence;
    
    if (isCleanLineArt) {
      imageType = 'Clean Line Art';
      pipeline = 'direct_trace';
      confidence = 'High';
    } else if (isPhotographic) {
      imageType = 'Photographic';
      pipeline = 'full_processing';
      confidence = 'Medium';
    } else if (hasComplexBackground) {
      imageType = 'Complex Pattern';
      pipeline = 'adaptive_background';
      confidence = 'High';
    } else {
      imageType = 'Standard Image';
      pipeline = 'balanced';
      confidence = 'Medium';
    }
    
    return {
      imageType,
      pipeline,
      confidence,
      stats: {
        backgroundRatio: (backgroundRatio * 100).toFixed(1),
        edgeRatio: (edgeRatio * 100).toFixed(1),
        contrast: stats.contrast.toFixed(1),
        brightness: stats.brightness.toFixed(1)
      },
      recommendations: {
        backgroundThreshold: isCleanLineArt ? 95 : hasComplexBackground ? 60 : 75,
        edgeSensitivity: isHighContrast ? 50 : isCleanLineArt ? 30 : 70,
        contourMinLength: isCleanLineArt ? 10 : 20,
        simplificationTolerance: isCleanLineArt ? 0.5 : hasComplexBackground ? 2.0 : 1.0
      }
    };
  }, []);

  const detectEdgePixels = (data, width, height) => {
    let edgeCount = 0;
    const threshold = 30;
    
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = (y * width + x) * 4;
        const current = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
        
        const neighbors = [
          0.299 * data[((y-1) * width + x) * 4] + 0.587 * data[((y-1) * width + x) * 4 + 1] + 0.114 * data[((y-1) * width + x) * 4 + 2],
          0.299 * data[(y * width + (x+1)) * 4] + 0.587 * data[(y * width + (x+1)) * 4 + 1] + 0.114 * data[(y * width + (x+1)) * 4 + 2],
          0.299 * data[((y+1) * width + x) * 4] + 0.587 * data[((y+1) * width + x) * 4 + 1] + 0.114 * data[((y+1) * width + x) * 4 + 2],
          0.299 * data[(y * width + (x-1)) * 4] + 0.587 * data[(y * width + (x-1)) * 4 + 1] + 0.114 * data[(y * width + (x-1)) * 4 + 2]
        ];
        
        const maxDiff = Math.max(...neighbors.map(n => Math.abs(current - n)));
        if (maxDiff > threshold) edgeCount++;
      }
    }
    
    return edgeCount;
  };

  // Simple edge detection for clean line art
  const simpleEdgeDetection = useCallback((imageData, threshold) => {
    const { data, width, height } = imageData;
    const result = new Uint8ClampedArray(data.length);
    
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = (y * width + x) * 4;
        const current = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
        
        const right = 0.299 * data[(y * width + (x+1)) * 4] + 0.587 * data[(y * width + (x+1)) * 4 + 1] + 0.114 * data[(y * width + (x+1)) * 4 + 2];
        const down = 0.299 * data[((y+1) * width + x) * 4] + 0.587 * data[((y+1) * width + x) * 4 + 1] + 0.114 * data[((y+1) * width + x) * 4 + 2];
        
        const gradient = Math.sqrt(Math.pow(right - current, 2) + Math.pow(down - current, 2));
        const value = gradient > threshold ? 255 : 0;
        
        result[idx] = value;
        result[idx + 1] = value;
        result[idx + 2] = value;
        result[idx + 3] = 255;
      }
    }
    
    return new ImageData(result, width, height);
  }, []);

  // Advanced Canny edge detection
  const cannyEdgeDetection = useCallback((imageData, lowThreshold, highThreshold, blurRadius) => {
    const { width, height } = imageData;
    const gray = new Float32Array(width * height);
    const result = new Uint8ClampedArray(imageData.data.length);
    
    // Convert to grayscale
    for (let i = 0; i < imageData.data.length; i += 4) {
      const grayValue = 0.299 * imageData.data[i] + 0.587 * imageData.data[i + 1] + 0.114 * imageData.data[i + 2];
      gray[i / 4] = grayValue;
    }
    
    const gradientMagnitude = new Float32Array(width * height);
    
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = y * width + x;
        
        const gx = -gray[(y-1)*width+(x-1)] + gray[(y-1)*width+(x+1)] +
                   -2*gray[y*width+(x-1)] + 2*gray[y*width+(x+1)] +
                   -gray[(y+1)*width+(x-1)] + gray[(y+1)*width+(x+1)];
        
        const gy = -gray[(y-1)*width+(x-1)] - 2*gray[(y-1)*width+x] - gray[(y-1)*width+(x+1)] +
                   gray[(y+1)*width+(x-1)] + 2*gray[(y+1)*width+x] + gray[(y+1)*width+(x+1)];
        
        gradientMagnitude[idx] = Math.sqrt(gx*gx + gy*gy);
      }
    }
    
    // Apply thresholds
    for (let i = 0; i < gradientMagnitude.length; i++) {
      const mag = gradientMagnitude[i];
      let value = 0;
      
      if (mag >= highThreshold) {
        value = 255;
      } else if (mag >= lowThreshold) {
        value = 128;
      }
      
      result[i * 4] = value;
      result[i * 4 + 1] = value;
      result[i * 4 + 2] = value;
      result[i * 4 + 3] = 255;
    }
    
    return new ImageData(result, width, height);
  }, []);

  // Background removal with Otsu's method
  const removeBackgroundOtsu = useCallback((imageData) => {
    const { data, width, height } = imageData;
    const histogram = new Array(256).fill(0);
    
    // Build histogram
    for (let i = 0; i < data.length; i += 4) {
      const gray = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
      histogram[gray]++;
    }
    
    // Otsu's method
    let sum = 0;
    for (let i = 0; i < 256; i++) sum += i * histogram[i];
    
    let sumB = 0, wB = 0, wF = 0;
    let varMax = 0;
    let threshold = 0;
    
    for (let t = 0; t < 256; t++) {
      wB += histogram[t];
      if (wB === 0) continue;
      
      wF = width * height - wB;
      if (wF === 0) break;
      
      sumB += t * histogram[t];
      const mB = sumB / wB;
      const mF = (sum - sumB) / wF;
      const varBetween = wB * wF * (mB - mF) * (mB - mF);
      
      if (varBetween > varMax) {
        varMax = varBetween;
        threshold = t;
      }
    }
    
    // Apply threshold
    const result = new Uint8ClampedArray(data.length);
    for (let i = 0; i < data.length; i += 4) {
      const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      const value = gray > threshold ? 255 : 0;
      
      result[i] = value;
      result[i + 1] = value;
      result[i + 2] = value;
      result[i + 3] = 255;
    }
    
    return new ImageData(result, width, height);
  }, []);

  // Advanced background removal
  const removeBackgroundAdvanced = useCallback((imageData, threshold) => {
    const data = new Uint8ClampedArray(imageData.data);
    
    for (let i = 0; i < imageData.data.length; i += 4) {
      const r = imageData.data[i];
      const g = imageData.data[i + 1];
      const b = imageData.data[i + 2];
      const gray = 0.299 * r + 0.587 * g + 0.114 * b;
      
      if (gray > threshold) {
        data[i] = 255;
        data[i + 1] = 255;
        data[i + 2] = 255;
      } else {
        const factor = 0.7;
        data[i] = Math.max(0, Math.min(255, r * factor));
        data[i + 1] = Math.max(0, Math.min(255, g * factor));
        data[i + 2] = Math.max(0, Math.min(255, b * factor));
      }
      data[i + 3] = 255;
    }
    
    return new ImageData(data, imageData.width, imageData.height);
  }, []);

  // Contour tracing
  const traceContoursOptimized = useCallback((edgeData, minLength) => {
    const { data, width, height } = edgeData;
    const visited = new Set();
    const contours = [];

    const isEdge = (x, y) => {
      if (x < 0 || x >= width || y < 0 || y >= height) return false;
      return data[(y * width + x) * 4] > 128;
    };

    const traceContour = (startX, startY) => {
      const contour = [];
      const directions = [[1,0], [1,1], [0,1], [-1,1], [-1,0], [-1,-1], [0,-1], [1,-1]];
      
      let x = startX, y = startY;
      let dir = 0;
      
      do {
        contour.push([x, y]);
        visited.add(`${x},${y}`);
        
        let found = false;
        for (let i = 0; i < 8; i++) {
          const newDir = (dir + i) % 8;
          const [dx, dy] = directions[newDir];
          const nx = x + dx;
          const ny = y + dy;
          
          if (isEdge(nx, ny) && !visited.has(`${nx},${ny}`)) {
            x = nx;
            y = ny;
            dir = (newDir + 6) % 8;
            found = true;
            break;
          }
        }
        
        if (!found) break;
        
      } while ((x !== startX || y !== startY) && contour.length < 10000);
      
      return contour;
    };

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (!visited.has(`${x},${y}`) && isEdge(x, y)) {
          const contour = traceContour(x, y);
          if (contour.length >= minLength) {
            contours.push(contour);
          }
        }
      }
    }
    
    return contours;
  }, []);

  // Douglas-Peucker simplification
  const douglasPeucker = (points, tolerance) => {
    if (points.length < 3) return points;
    
    const simplifyRecursive = (pts, tol) => {
      if (pts.length < 3) return pts;
      
      let maxDist = 0;
      let index = 0;
      
      for (let i = 1; i < pts.length - 1; i++) {
        const dist = distanceToLine(pts[i], pts[0], pts[pts.length - 1]);
        if (dist > maxDist) {
          index = i;
          maxDist = dist;
        }
      }
      
      if (maxDist > tol) {
        const left = simplifyRecursive(pts.slice(0, index + 1), tol);
        const right = simplifyRecursive(pts.slice(index), tol);
        return [...left.slice(0, -1), ...right];
      }
      
      return [pts[0], pts[pts.length - 1]];
    };
    
    return simplifyRecursive(points, tolerance);
  };

  const distanceToLine = (point, lineStart, lineEnd) => {
    const [px, py] = point;
    const [x1, y1] = lineStart;
    const [x2, y2] = lineEnd;
    
    const A = px - x1;
    const B = py - y1;
    const C = x2 - x1;
    const D = y2 - y1;
    
    const dot = A * C + B * D;
    const lenSq = C * C + D * D;
    
    if (lenSq === 0) return Math.sqrt(A * A + B * B);
    
    let param = dot / lenSq;
    param = Math.max(0, Math.min(1, param));
    
    const xx = x1 + param * C;
    const yy = y1 + param * D;
    
    const dx = px - xx;
    const dy = py - yy;
    
    return Math.sqrt(dx * dx + dy * dy);
  };

  const calculateContourArea = (contour) => {
    if (contour.length < 3) return 0;
    let area = 0;
    for (let i = 0; i < contour.length; i++) {
      const j = (i + 1) % contour.length;
      area += contour[i][0] * contour[j][1];
      area -= contour[j][0] * contour[i][1];
    }
    return Math.abs(area) / 2;
  };

  // Processing pipelines
  const processDirectTrace = useCallback(async (imageData) => {
    const edges = simpleEdgeDetection(imageData, 20);
    const contours = traceContoursOptimized(edges, 10);
    return contours.map(contour => douglasPeucker(contour, 0.5));
  }, []);

  const processFullProcessing = useCallback(async (imageData, analysis) => {
    const backgroundRemoved = removeBackgroundAdvanced(imageData, analysis.recommendations.backgroundThreshold);
    const edges = cannyEdgeDetection(backgroundRemoved, analysis.recommendations.edgeSensitivity * 0.4, analysis.recommendations.edgeSensitivity * 1.2, 1);
    const contours = traceContoursOptimized(edges, analysis.recommendations.contourMinLength);
    return contours.map(contour => douglasPeucker(contour, analysis.recommendations.simplificationTolerance)).filter(contour => calculateContourArea(contour) > 100);
  }, []);

  const processAdaptiveBackground = useCallback(async (imageData, analysis) => {
    const backgroundRemoved = removeBackgroundOtsu(imageData);
    const edges = cannyEdgeDetection(backgroundRemoved, analysis.recommendations.edgeSensitivity * 0.5, analysis.recommendations.edgeSensitivity * 1.0, 2);
    const contours = traceContoursOptimized(edges, analysis.recommendations.contourMinLength);
    return contours.map(contour => douglasPeucker(contour, analysis.recommendations.simplificationTolerance)).filter(contour => calculateContourArea(contour) > 50);
  }, []);

  const processBalanced = useCallback(async (imageData, analysis) => {
    const backgroundRemoved = removeBackgroundAdvanced(imageData, analysis.recommendations.backgroundThreshold);
    const edges = cannyEdgeDetection(backgroundRemoved, analysis.recommendations.edgeSensitivity * 0.6, analysis.recommendations.edgeSensitivity * 1.0, 1);
    const contours = traceContoursOptimized(edges, analysis.recommendations.contourMinLength);
    return contours.map(contour => douglasPeucker(contour, analysis.recommendations.simplificationTolerance)).filter(contour => calculateContourArea(contour) > 50);
  }, []);

  // Production-grade DXF generator
  const generateProductionDXF = useCallback((contours, imageWidth, imageHeight) => {
    try {
      const { width, height, units, precision } = config;
      const scaleX = width / imageWidth;
      const scaleY = height / imageHeight;
      
      let dxf = [];
      let handle = 100;
      
      // Professional DXF header
      dxf.push('0\nSECTION\n2\nHEADER');
      dxf.push('9\n$ACADVER\n1\nAC1027');
      dxf.push('9\n$INSBASE\n10\n0\n20\n0\n30\n0');
      dxf.push('9\n$EXTMIN\n10\n0\n20\n0\n30\n0');
      dxf.push('9\n$EXTMAX\n10\n' + width + '\n20\n' + height + '\n30\n0');
      dxf.push('9\n$MEASUREMENT\n70\n' + (units === 'mm' ? '1' : '0'));
      dxf.push('0\nENDSEC');
      
      // Tables section
      dxf.push('0\nSECTION\n2\nTABLES');
      dxf.push('0\nTABLE\n2\nLAYER\n70\n2');
      dxf.push('0\nLAYER\n2\nCUT_PATH\n70\n0\n62\n1\n6\nCONTINUOUS');
      dxf.push('0\nLAYER\n2\nCONSTRUCTION\n70\n0\n62\n8\n6\nDASHED');
      dxf.push('0\nENDTAB\n0\nENDSEC');
      
      dxf.push('0\nSECTION\n2\nBLOCKS\n0\nENDSEC');
      dxf.push('0\nSECTION\n2\nENTITIES');
      
      // Add border rectangle
      dxf.push('0\nLWPOLYLINE');
      dxf.push('5\n' + handle.toString(16).toUpperCase());
      dxf.push('100\nAcDbEntity\n8\nCONSTRUCTION');
      dxf.push('100\nAcDbPolyline\n90\n4\n70\n1');
      dxf.push('10\n0\n20\n0');
      dxf.push('10\n' + width + '\n20\n0');
      dxf.push('10\n' + width + '\n20\n' + height);
      dxf.push('10\n0\n20\n' + height);
      handle++;
      
      const decimalPlaces = precision === 'high' ? 3 : precision === 'medium' ? 2 : 1;
      
      // Add contours
      contours.forEach((contour, index) => {
        if (contour.length < 2) return;
        
        dxf.push('0\nLWPOLYLINE');
        dxf.push('5\n' + handle.toString(16).toUpperCase());
        dxf.push('100\nAcDbEntity\n8\nCUT_PATH');
        dxf.push('62\n' + (index % 7 + 1));
        dxf.push('100\nAcDbPolyline');
        dxf.push('90\n' + contour.length);
        dxf.push('70\n0');
        
        contour.forEach(point => {
          const x = (point[0] * scaleX).toFixed(decimalPlaces);
          const y = (height - point[1] * scaleY).toFixed(decimalPlaces);
          dxf.push('10\n' + x + '\n20\n' + y);
        });
        
        handle++;
      });
      
      dxf.push('0\nENDSEC\n0\nEOF');
      return dxf.join('\n');
      
    } catch (error) {
      throw new Error('DXF Generation Error: ' + error.message);
    }
  }, [config]);

  // Main processing function
  const processImageIntelligently = async () => {
    if (!image || !imageAnalysis) return;
    
    setProcessing(true);
    setProgress(0);
    setErrors([]);
    let processedContours = [];
    let processedImageData = null;
    let edgesImageData = null;
    
    try {
      const canvas = canvasRefs.working.current;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      
      if (rotation === 90 || rotation === 270) {
        canvas.width = image.height;
        canvas.height = image.width;
      } else {
        canvas.width = image.width;
        canvas.height = image.height;
      }
      
      ctx.save();
      ctx.translate(canvas.width / 2, canvas.height / 2);
      ctx.rotate((rotation * Math.PI) / 180);
      ctx.drawImage(image, -image.width / 2, -image.height / 2);
      ctx.restore();
      
      const originalData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      
      setCurrentStep('Processing as ' + imageAnalysis.imageType + '...');
      setProgress(20);
      await new Promise(resolve => setTimeout(resolve, 100));
      
      // Step 1: Background processing (show intermediate result)
      setCurrentStep('Background processing...');
      setProgress(30);
      await new Promise(resolve => setTimeout(resolve, 100));
      
      if (imageAnalysis.pipeline === 'direct_trace') {
        processedImageData = originalData; // No background removal needed
      } else {
        processedImageData = imageAnalysis.pipeline === 'adaptive_background' 
          ? removeBackgroundOtsu(originalData)
          : removeBackgroundAdvanced(originalData, imageAnalysis.recommendations.backgroundThreshold);
      }
      
      // Update processed canvas
      const processedCanvas = canvasRefs.processed.current;
      processedCanvas.width = canvas.width;
      processedCanvas.height = canvas.height;
      processedCanvas.getContext('2d').putImageData(processedImageData, 0, 0);
      
      // Step 2: Edge detection (show intermediate result)
      setCurrentStep('Edge detection...');
      setProgress(50);
      await new Promise(resolve => setTimeout(resolve, 100));
      
      if (imageAnalysis.pipeline === 'direct_trace') {
        edgesImageData = simpleEdgeDetection(processedImageData, 20);
      } else {
        const lowThresh = imageAnalysis.recommendations.edgeSensitivity * 0.4;
        const highThresh = imageAnalysis.recommendations.edgeSensitivity * 1.2;
        edgesImageData = cannyEdgeDetection(processedImageData, lowThresh, highThresh, 1);
      }
      
      // Update edges canvas
      const edgesCanvas = canvasRefs.edges.current;
      edgesCanvas.width = canvas.width;
      edgesCanvas.height = canvas.height;
      edgesCanvas.getContext('2d').putImageData(edgesImageData, 0, 0);
      
      // Step 3: Contour tracing
      setCurrentStep('Tracing contours...');
      setProgress(70);
      await new Promise(resolve => setTimeout(resolve, 100));
      
      const rawContours = traceContoursOptimized(edgesImageData, imageAnalysis.recommendations.contourMinLength);
      
      // Step 4: Simplification based on pipeline
      processedContours = rawContours.map(contour => 
        douglasPeucker(contour, imageAnalysis.recommendations.simplificationTolerance)
      ).filter(contour => {
        const area = calculateContourArea(contour);
        return imageAnalysis.pipeline === 'direct_trace' ? area > 25 : area > 50;
      });
      
      setProgress(90);
      setCurrentStep('Generating high-quality DXF...');
      await new Promise(resolve => setTimeout(resolve, 100));
      
      const dxf = generateProductionDXF(processedContours, canvas.width, canvas.height);
      
      setCurrentStep('Finalizing output...');
      drawToolpathPreview(processedContours, canvas.width, canvas.height);
      
      setResults({
        contours: processedContours,
        dxf,
        stats: {
          contoursFound: processedContours.length,
          totalPoints: processedContours.reduce((sum, c) => sum + c.length, 0),
          processingPipeline: imageAnalysis.pipeline,
          confidence: imageAnalysis.confidence,
          fileSize: Math.round(dxf.length / 1024)
        }
      });
      
      setCurrentStep('Processing complete!');
      setProgress(100);
      
    } catch (error) {
      console.error('Processing error:', error);
      setErrors([error.message]);
      setCurrentStep('Processing failed');
    } finally {
      setTimeout(() => setProcessing(false), 1000);
    }
  };

  const drawToolpathPreview = (contours, imageWidth, imageHeight) => {
    const canvas = canvasRefs.preview.current;
    const ctx = canvas.getContext('2d');
    
    canvas.width = 400;
    canvas.height = 300;
    
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    const scaleX = canvas.width / imageWidth;
    const scaleY = canvas.height / imageHeight;
    const scale = Math.min(scaleX, scaleY) * 0.9;
    const offsetX = (canvas.width - imageWidth * scale) / 2;
    const offsetY = (canvas.height - imageHeight * scale) / 2;
    
    // Draw grid
    ctx.strokeStyle = '#f0f0f0';
    ctx.lineWidth = 1;
    for (let x = 0; x <= canvas.width; x += 20) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }
    for (let y = 0; y <= canvas.height; y += 20) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }
    
    // Draw contours
    contours.forEach((contour, i) => {
      if (contour.length < 2) return;
      
      const area = calculateContourArea(contour);
      const isLarge = area > 500;
      const isMedium = area > 100;
      
      ctx.strokeStyle = isLarge ? '#e74c3c' : isMedium ? '#f39c12' : '#3498db';
      ctx.lineWidth = isLarge ? 2 : 1;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      
      ctx.beginPath();
      ctx.moveTo(
        contour[0][0] * scale + offsetX,
        contour[0][1] * scale + offsetY
      );
      
      for (let j = 1; j < contour.length; j++) {
        ctx.lineTo(
          contour[j][0] * scale + offsetX,
          contour[j][1] * scale + offsetY
        );
      }
      
      ctx.stroke();
      
      ctx.fillStyle = ctx.strokeStyle;
      ctx.beginPath();
      ctx.arc(
        contour[0][0] * scale + offsetX,
        contour[0][1] * scale + offsetY,
        2, 0, Math.PI * 2
      );
      ctx.fill();
    });
    
    // Draw stats
    ctx.fillStyle = '#2c3e50';
    ctx.font = 'bold 12px Arial';
    ctx.fillText(contours.length + ' Contours', 10, 20);
    ctx.fillText(contours.reduce((sum, c) => sum + c.length, 0) + ' Points', 10, 35);
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    if (!file.type.startsWith('image/')) {
      setErrors(['Please select a valid image file']);
      return;
    }
    
    if (file.size > 10 * 1024 * 1024) {
      setErrors(['Image file too large (max 10MB)']);
      return;
    }
    
    const reader = new FileReader();
    reader.onload = (event) => {
      const img = new Image();
      img.onload = () => {
        setImage(img);
        setResults(null);
        setErrors([]);
        
        const canvas = canvasRefs.original.current;
        if (canvas) {
          const ctx = canvas.getContext('2d');
          canvas.width = img.width;
          canvas.height = img.height;
          ctx.drawImage(img, 0, 0);
          
          const imageData = ctx.getImageData(0, 0, img.width, img.height);
          const analysis = analyzeImageIntelligently(imageData);
          setImageAnalysis(analysis);
          
          setConfig(prev => ({
            ...prev,
            processingPipeline: analysis.pipeline,
            backgroundThreshold: analysis.recommendations.backgroundThreshold,
            edgeSensitivity: analysis.recommendations.edgeSensitivity,
            contourMinLength: analysis.recommendations.contourMinLength
          }));
        }
      };
      img.src = event.target.result;
    };
    reader.readAsDataURL(file);
  };

  const downloadFile = (content, filename) => {
    try {
      const blob = new Blob([content], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      setErrors(['Download failed: ' + error.message]);
    }
  };

  const handleRotate = () => {
    setRotation((prev) => (prev + 90) % 360);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="max-w-7xl mx-auto p-6">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <Zap className="w-12 h-12 text-yellow-400 mr-4" />
            <h1 className="text-5xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
              SUPER ALIEN PRO
            </h1>
            <Cpu className="w-12 h-12 text-cyan-400 ml-4" />
          </div>
          <p className="text-xl text-gray-300 mb-2">Production-Grade Intelligent DXF Converter</p>
          <div className="flex justify-center space-x-6 text-sm text-gray-400">
            <span className="flex items-center"><Target className="w-4 h-4 mr-1" />Smart Analysis</span>
            <span className="flex items-center"><Eye className="w-4 h-4 mr-1" />Auto Pipeline Selection</span>
            <span className="flex items-center"><Cpu className="w-4 h-4 mr-1" />Production Quality</span>
          </div>
        </div>

        {/* Error Display */}
        {errors.length > 0 && (
          <div className="bg-red-900/50 border border-red-400/50 rounded-xl p-4 mb-6">
            <div className="flex items-center mb-2">
              <AlertTriangle className="w-5 h-5 text-red-400 mr-2" />
              <h3 className="text-red-300 font-semibold">Processing Errors</h3>
            </div>
            {errors.map((error, idx) => (
              <p key={idx} className="text-red-200 text-sm">{error}</p>
            ))}
          </div>
        )}

        {/* Upload Section */}
        <div className="bg-gradient-to-r from-slate-800 to-slate-700 rounded-2xl shadow-2xl p-8 mb-8 border border-cyan-500/20">
          <div className="text-center">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="hidden"
            />
            <div
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-cyan-400 rounded-xl p-12 cursor-pointer hover:border-purple-400 transition-all duration-300 group"
            >
              <Upload className="w-16 h-16 mx-auto text-cyan-400 group-hover:text-purple-400 mb-4 transition-colors" />
              <h3 className="text-2xl font-bold text-white mb-2">Upload Your Image</h3>
              <p className="text-gray-400">AI will analyze and select optimal processing pipeline</p>
            </div>
          </div>
        </div>

        {/* Image Analysis Results */}
        {imageAnalysis && (
          <div className="bg-gradient-to-r from-emerald-900/50 to-emerald-800/30 rounded-xl p-6 mb-8 border border-emerald-400/20">
            <div className="flex items-center mb-4">
              <CheckCircle className="w-6 h-6 text-emerald-400 mr-3" />
              <h3 className="text-xl font-semibold text-emerald-300">Intelligent Analysis Complete</h3>
            </div>
            
            <div className="grid md:grid-cols-3 gap-6">
              <div className="bg-slate-800/50 rounded-lg p-4">
                <h4 className="text-emerald-200 font-semibold mb-2">Image Classification</h4>
                <p className="text-white text-lg">{imageAnalysis.imageType}</p>
                <p className="text-emerald-300 text-sm">Confidence: {imageAnalysis.confidence}</p>
              </div>
              
              <div className="bg-slate-800/50 rounded-lg p-4">
                <h4 className="text-emerald-200 font-semibold mb-2">Processing Pipeline</h4>
                <p className="text-white text-lg">{imageAnalysis.pipeline.replace('_', ' ')}</p>
                <p className="text-emerald-300 text-sm">Auto-selected optimal method</p>
              </div>
              
              <div className="bg-slate-800/50 rounded-lg p-4">
                <h4 className="text-emerald-200 font-semibold mb-2">Image Statistics</h4>
                <div className="text-sm text-gray-300">
                  <p>Background: {imageAnalysis.stats.backgroundRatio}%</p>
                  <p>Edges: {imageAnalysis.stats.edgeRatio}%</p>
                  <p>Contrast: {imageAnalysis.stats.contrast}</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Configuration Panel */}
        <div className="grid lg:grid-cols-3 gap-6 mb-8">
          <div className="bg-gradient-to-br from-blue-900/50 to-blue-800/30 rounded-xl p-6 border border-blue-400/20">
            <h3 className="text-lg font-semibold text-blue-300 mb-4 flex items-center">
              <Settings className="w-5 h-5 mr-2" />Target Dimensions
            </h3>
            <div className="space-y-3">
              <input
                type="number"
                value={config.width}
                onChange={(e) => setConfig(prev => ({...prev, width: Number(e.target.value)}))}
                className="w-full bg-slate-700 border border-blue-400/30 rounded-lg px-3 py-2 text-white"
                placeholder="Width"
              />
              <input
                type="number"
                value={config.height}
                onChange={(e) => setConfig(prev => ({...prev, height: Number(e.target.value)}))}
                className="w-full bg-slate-700 border border-blue-400/30 rounded-lg px-3 py-2 text-white"
                placeholder="Height"
              />
              <select
                value={config.units}
                onChange={(e) => setConfig(prev => ({...prev, units: e.target.value}))}
                className="w-full bg-slate-700 border border-blue-400/30 rounded-lg px-3 py-2 text-white"
              >
                <option value="mm">Millimeters</option>
                <option value="inch">Inches</option>
              </select>
              <button
                onClick={handleRotate}
                className="w-full bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded-lg transition-colors flex items-center justify-center"
              >
                <RotateCw className="w-4 h-4 mr-2" />
                Rotate {rotation}°
              </button>
            </div>
          </div>

          <div className="bg-gradient-to-br from-purple-900/50 to-purple-800/30 rounded-xl p-6 border border-purple-400/20">
            <h3 className="text-lg font-semibold text-purple-300 mb-4">Quality Control</h3>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-purple-200 block mb-1">Precision Level</label>
                <select
                  value={config.precision}
                  onChange={(e) => setConfig(prev => ({...prev, precision: e.target.value}))}
                  className="w-full bg-slate-700 border border-purple-400/30 rounded-lg px-3 py-2 text-white"
                >
                  <option value="high">High (0.001 precision)</option>
                  <option value="medium">Medium (0.01 precision)</option>
                  <option value="low">Low (0.1 precision)</option>
                </select>
              </div>
              
              <div>
                <label className="text-xs text-purple-200 block mb-1">Contour Simplification</label>
                <select
                  value={config.simplification}
                  onChange={(e) => setConfig(prev => ({...prev, simplification: e.target.value}))}
                  className="w-full bg-slate-700 border border-purple-400/30 rounded-lg px-3 py-2 text-white"
                >
                  <option value="auto">Auto (AI Optimized)</option>
                  <option value="minimal">Minimal (Max Detail)</option>
                  <option value="aggressive">Aggressive (Smaller Files)</option>
                </select>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-br from-green-900/50 to-green-800/30 rounded-xl p-6 border border-green-400/20">
            <h3 className="text-lg font-semibold text-green-300 mb-4">Processing Status</h3>
            <div className="space-y-3">
              {imageAnalysis ? (
                <>
                  <div className="text-sm text-green-200">
                    <p><strong>Method:</strong> {imageAnalysis.pipeline}</p>
                    <p><strong>Threshold:</strong> {config.backgroundThreshold}%</p>
                    <p><strong>Edge Sens.:</strong> {config.edgeSensitivity}</p>
                  </div>
                  <div className="bg-green-800/30 rounded-lg p-3">
                    <p className="text-green-200 text-xs">Auto-optimized settings applied based on image analysis</p>
                  </div>
                </>
              ) : (
                <p className="text-gray-400 text-sm">Upload an image to see analysis</p>
              )}
            </div>
          </div>
        </div>

        {/* Process Button */}
        <div className="text-center mb-8">
          <button
            onClick={processImageIntelligently}
            disabled={!image || !imageAnalysis || processing}
            className="bg-gradient-to-r from-cyan-600 to-purple-600 hover:from-cyan-500 hover:to-purple-500 disabled:from-gray-600 disabled:to-gray-600 text-white px-12 py-4 rounded-xl font-bold text-lg shadow-2xl transition-all duration-300 flex items-center mx-auto"
          >
            <Play className="w-6 h-6 mr-3" />
            {processing ? 'Processing...' : imageAnalysis ? 'GENERATE PRODUCTION DXF' : 'Upload Image First'}
          </button>
          
          {processing && (
            <div className="mt-6 max-w-md mx-auto">
              <div className="bg-slate-700 rounded-full h-4 mb-2 overflow-hidden">
                <div 
                  className="bg-gradient-to-r from-cyan-400 to-purple-400 h-4 rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className="text-cyan-300 font-semibold">{currentStep}</p>
            </div>
          )}
        </div>

        {/* Results - Complete Processing Pipeline */}
        {image && (
          <div className="bg-gradient-to-r from-slate-800 to-slate-700 rounded-2xl p-8 shadow-2xl border border-cyan-500/20 mb-8">
            <h2 className="text-2xl font-bold text-white mb-6 text-center">Complete Processing Pipeline</h2>
            
            <div className="grid lg:grid-cols-4 gap-6">
              <div className="text-center">
                <h3 className="text-lg font-semibold text-gray-300 mb-3">Original Image</h3>
                <canvas
                  ref={canvasRefs.original}
                  className="w-full h-48 object-contain bg-slate-600 rounded-lg border border-gray-500"
                />
              </div>
              
              <div className="text-center">
                <h3 className="text-lg font-semibold text-green-300 mb-3">Background Processed</h3>
                <canvas
                  ref={canvasRefs.processed}
                  className="w-full h-48 object-contain bg-slate-600 rounded-lg border border-green-400"
                />
              </div>
              
              <div className="text-center">
                <h3 className="text-lg font-semibold text-orange-300 mb-3">Edge Detection</h3>
                <canvas
                  ref={canvasRefs.edges}
                  className="w-full h-48 object-contain bg-slate-600 rounded-lg border border-orange-400"
                />
              </div>
              
              <div className="text-center">
                <h3 className="text-lg font-semibold text-purple-300 mb-3">DXF Preview</h3>
                <canvas
                  ref={canvasRefs.preview}
                  className="w-full h-48 object-contain bg-white rounded-lg border border-purple-400"
                />
              </div>
            </div>
          </div>
        )}

        {/* Download Section */}
        {results && (
          <div className="bg-gradient-to-r from-emerald-800 to-emerald-700 rounded-2xl p-8 shadow-2xl border border-emerald-400/20">
            <h3 className="text-2xl font-bold text-emerald-200 mb-6 text-center">
              PRODUCTION DXF READY
            </h3>
            
            <div className="text-center mb-6">
              <button
                onClick={() => downloadFile(results.dxf, `production_${config.width}x${config.height}_${Date.now()}.dxf`)}
                className="bg-gradient-to-r from-green-600 to-green-500 hover:from-green-500 hover:to-green-400 text-white py-4 px-8 rounded-xl font-bold text-lg shadow-lg transition-all duration-300 flex items-center mx-auto"
              >
                <Download className="w-6 h-6 mr-3" />
                Download Production DXF ({results.stats.fileSize}KB)
              </button>
            </div>

            <div className="bg-slate-800/50 rounded-xl p-6 border border-emerald-400/20">
              <div className="grid md:grid-cols-4 gap-6 text-center">
                <div>
                  <span className="text-emerald-300 font-bold text-2xl">{results.stats.contoursFound}</span>
                  <p className="text-gray-400">Contours</p>
                </div>
                <div>
                  <span className="text-emerald-300 font-bold text-2xl">{results.stats.totalPoints}</span>
                  <p className="text-gray-400">Points</p>
                </div>
                <div>
                  <span className="text-emerald-300 font-bold text-2xl">{results.stats.confidence}</span>
                  <p className="text-gray-400">Confidence</p>
                </div>
                <div>
                  <span className="text-emerald-300 font-bold text-2xl">{results.stats.processingPipeline}</span>
                  <p className="text-gray-400">Pipeline</p>
                </div>
              </div>
              
              <div className="mt-6 text-center">
                <p className="text-emerald-200 text-sm">
                  Processed using {results.stats.processingPipeline} pipeline with {results.stats.confidence.toLowerCase()} confidence
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Hidden working canvas */}
        <canvas ref={canvasRefs.working} style={{ display: 'none' }} />
      </div>
    </div>
  );
};

export default ProductionImageConverter;// Optimized contour tracing using Moore-Neighbor tracing algorithm