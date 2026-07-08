import { useEffect, useRef } from "react";

function Canvas({ width, height, brushSize }: { width: number, height: number, brushSize: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const isDrawing = useRef(false);
  const lastPos = useRef({ x: 0, y: 0 });

  // context only exists once the canvas is mounted, so this cant run at render time
  useEffect(() => {
    const context = canvasRef.current?.getContext("2d");
    if (!context) return;

    context.lineCap = "round";
    context.lineJoin = "round";
    context.strokeStyle = "black";
    context.lineWidth = brushSize;
  }, [brushSize]);

  const getPos = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    isDrawing.current = true;
    lastPos.current = getPos(e);
  };

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!isDrawing.current) return;
    const context = canvasRef.current?.getContext("2d");
    if (!context) return;

    const pos = getPos(e);
    context.beginPath();
    context.moveTo(lastPos.current.x, lastPos.current.y);
    context.lineTo(pos.x, pos.y);
    context.stroke();
    lastPos.current = pos;
  };

  const onPointerUp = () => {
    isDrawing.current = false;
  };

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerLeave={onPointerUp}
      className="border border-neutral-700 rounded-md touch-none"
    />
  );
};

export default function() {
  return (
    <div>
      <Canvas width={280} height={280} brushSize={12} />
    </div>
  )
};