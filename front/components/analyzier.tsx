"use client";

import type React from "react";
import { useState } from "react";
import { Card, CardHeader, CardBody, CardFooter } from "@heroui/card";
import { Button, ButtonGroup } from "@heroui/button";
import {
  Upload,
  ImageIcon,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Activity,
} from "lucide-react";
import Image from "next/image";

const API_URL = process.env.NEXT_PUBLIC_BASE_URL;

interface AnalysisResult {
  segmentation_map: string;
  viability: {
    status: string;
    message: string;
    composition: {
      suelo: number;
      arena: number;
      rocas: number;
    };
  };
}

export default function Analyzier() {
  const [image, setImage] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImage(reader.result as string);
        setResults(null);
        setError(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const analyzeImage = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        console.log(errorData)
        const msg =
          errorData?.detail || "Ocurrió un error desconocido en el servidor.";
        throw new Error(msg);
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      console.error("Error capturado:", err);
      const errorMessage = err instanceof Error ? err.message : String(err);
      if (errorMessage === "Failed to fetch") {
        setError(
          "No se puede conectar con el servidor. Verifica que el backend esté encendido."
        );
      } else {
        setError(errorMessage);
      }
    } finally {
      setIsAnalyzing(false);
    }
  };

  // return (
  //   <div className="grid gap-4 md:grid-cols-2">
  //     <Card className="bg-card border gap-4 p-6 mx-8 mt-10 rounded-xl">
  //       <CardHeader className="flex-col items-start gap-2">
  //         <h1 className="font-semibold">Cargar imagen</h1>
  //         <p className="text-muted-foreground">
  //           Cargue una imagen para comenzar el análisis del terreno.
  //         </p>
  //       </CardHeader>
  //       <CardBody className="space-y-4">
  //         {!image ? (
  //           <label className="flex min-h-100 cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-border bg-muted/30 transition-colors hover:bg-muted/50">
  //             <Upload className="h-12 w-12 text-muted-foreground" />
  //             <p className="mt-2 text-sm font-medium text-foreground">
  //               Carga de imagen
  //             </p>
  //             <p className="mt-1 text-xs text-muted-foreground">
  //               PNG, JPG +10MB
  //             </p>
  //             <input
  //               type="file"
  //               className="hidden"
  //               accept="image/*"
  //               onChange={handleImageUpload}
  //             />
  //           </label>
  //         ) : (
  //           <div className="space-y-4">
  //             <div className="relative overflow-hidden rounded-lg border border-border">
  //               <Image
  //                 width={75}
  //                 height={75}
  //                 src={image || "/placeholder.svg"}
  //                 alt="Uploaded Mars terrain"
  //                 className="h-auto w-full object-cover"
  //               />
  //             </div>
  //             <div className="flex gap-2">
  //               <Button
  //                 onClick={analyzeImage}
  //                 color="primary"
  //                 fullWidth
  //                 disabled={isAnalyzing}
  //                 className="flex items-center rounded-md"
  //               >
  //                 {isAnalyzing ? (
  //                   <>
  //                     <span className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
  //                     Analizando...
  //                   </>
  //                 ) : (
  //                   <>
  //                     <ImageIcon className="mr-2 h-4 w-4" />
  //                     Analizar Terreno
  //                   </>
  //                 )}
  //               </Button>
  //               <Button
  //                 variant="bordered"
  //                 onClick={() => {
  //                   setImage(null);
  //                   //setResults(null);
  //                 }}
  //                 className="flex items-center rounded-md border-2"
  //               >
  //                 Limpiar
  //               </Button>
  //             </div>
  //           </div>
  //         )}
  //       </CardBody>
  //     </Card>

  //     <Card className="bg-card border gap-4 p-6 mx-8 mt-10 rounded-xl">
  //       <CardHeader className="flex-col items-start gap-2">
  //         <h1 className="font-semibold">Análisis de resultados</h1>
  //         <p className="text-muted-foreground">
  //           Resultados de la caracterización del terreno y la evaluación de la
  //           transitabilidad.
  //         </p>
  //       </CardHeader>
  //       <CardBody>
  //         {!results ? (
  //           <div className="flex min-h-100 flex-col items-center justify-center text-center">
  //             <div className="rounded-full bg-muted p-4">
  //               <ImageIcon className="h-8 w-8 text-muted-foreground" />
  //             </div>
  //             <p className="mt-4 text-sm font-medium text-foreground">
  //               Aún no hay análisis.
  //             </p>
  //             <p className="mt-1 text-xs text-muted-foreground">
  //               Cargue una imagen y haga clic en "Analizar Terreno" para ver los
  //               resultados.
  //             </p>
  //           </div>
  //         ) : (
  //           <div></div>
  //         )}
  //       </CardBody>
  //     </Card>
  //   </div>
  // );

  return (
    <div className="grid gap-4 md:grid-cols-2">
      {/* TARJETA 1: CARGA DE IMAGEN */}
      <Card className="bg-card border gap-4 p-6 mx-8 mt-10 rounded-xl">
        <CardHeader className="flex-col items-start gap-2">
          <h1 className="font-semibold text-lg">Cargar imagen del Rover</h1>
          <p className="text-muted-foreground text-sm">
            Cargue una imagen capturada por el sistema de cámaras para evaluar
            el terreno.
          </p>
        </CardHeader>
        <CardBody className="space-y-4">
          {!image ? (
            <label className="flex min-h-[250px] cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-border bg-muted/30 transition-colors hover:bg-muted/50">
              <Upload className="h-12 w-12 text-muted-foreground mb-3" />
              <p className="text-sm font-medium text-foreground">
                Click para cargar imagen
              </p>
              <p className="mt-1 text-xs text-muted-foreground">
                PNG, JPG (Máx 10MB)
              </p>
              <input
                type="file"
                className="hidden"
                accept="image/*"
                onChange={handleImageUpload}
              />
            </label>
          ) : (
            <div className="space-y-4">
              <div className="relative overflow-hidden rounded-lg border border-border bg-black/5">
                <Image
                  width={400}
                  height={300}
                  src={image || "/placeholder.svg"}
                  alt="Uploaded Mars terrain"
                  className="h-64 w-full object-contain"
                />
              </div>

              {error && (
                <div className="p-3 bg-red-100/10 border border-red-500/50 rounded-md flex items-center gap-2 text-red-500 text-sm">
                  <XCircle className="h-4 w-4" />
                  {error}
                </div>
              )}

              <div className="flex gap-2">
                <Button
                  onClick={analyzeImage}
                  color="primary"
                  fullWidth
                  disabled={isAnalyzing}
                  className="flex items-center rounded-md font-semibold"
                >
                  {isAnalyzing ? (
                    <>
                      <span className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                      Procesando Topografía...
                    </>
                  ) : (
                    <>
                      <Activity className="mr-2 h-4 w-4" />
                      Analizar Terreno
                    </>
                  )}
                </Button>
                <Button
                  variant="bordered"
                  onClick={() => {
                    setImage(null);
                    setResults(null);
                    setSelectedFile(null);
                    setError(null);
                  }}
                  className="flex items-center rounded-md border-2"
                >
                  Limpiar
                </Button>
              </div>
            </div>
          )}
        </CardBody>
      </Card>

      {/* TARJETA 2: RESULTADOS */}
      <Card className="bg-card border gap-4 p-6 mx-8 mt-10 rounded-xl">
        <CardHeader className="flex-col items-start gap-2">
          <h1 className="font-semibold text-lg">Análisis de resultados</h1>
          <p className="text-muted-foreground text-sm">
            Segmentación semántica y evaluación de transitabilidad generada por
            U-Net.
          </p>
        </CardHeader>
        <CardBody className="h-full">
          {!results ? (
            <div className="flex h-full flex-col items-center justify-center text-center opacity-60 min-h-[200px]">
              <div className="rounded-full bg-muted p-4 mb-4">
                <ImageIcon className="h-8 w-8 text-muted-foreground" />
              </div>
              <p className="text-sm font-medium text-foreground">
                Esperando datos del sensor...
              </p>
              <p className="mt-1 text-xs text-muted-foreground max-w-[250px]">
                El modelo de IA procesará la imagen para identificar rocas,
                arena y suelo firme.
              </p>
            </div>
          ) : (
            <div className="space-y-6 animate-in fade-in duration-500">
              {/* 1. MÁSCARA DE SEGMENTACIÓN */}
              <div className="relative overflow-hidden rounded-lg border-2 border-primary/20 shadow-sm">
                <img
                  src={results.segmentation_map}
                  alt="Mapa de Segmentación"
                  className="w-full h-64 object-contain bg-black"
                />
                <div className="absolute bottom-2 left-2 right-2 bg-black/70 backdrop-blur-md rounded-md p-2 flex justify-between text-[10px] text-white">
                  <div className="flex items-center gap-1">
                    <div className="w-2 h-2 rounded-full bg-[#808080]"></div>
                    Suelo
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-2 h-2 rounded-full bg-[#E6C800]"></div>
                    Arena
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-2 h-2 rounded-full bg-[#C83232]"></div>
                    Roca
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-2 h-2 rounded-full bg-[#00C800]"></div>
                    Obs.
                  </div>
                </div>
              </div>

              {/* 2. ESTADO DE VIABILIDAD */}
              <div
                className={`p-4 rounded-lg border-l-4 ${
                  results.viability.status === "VIABLE"
                    ? "bg-green-500/10 border-green-500"
                    : results.viability.status === "PELIGRO"
                    ? "bg-red-500/10 border-red-500"
                    : "bg-yellow-500/10 border-yellow-500"
                }`}
              >
                <div className="flex items-center gap-2 mb-1">
                  {results.viability.status === "VIABLE" ? (
                    <CheckCircle2 className="text-green-500 h-5 w-5" />
                  ) : results.viability.status === "PELIGRO" ? (
                    <XCircle className="text-red-500 h-5 w-5" />
                  ) : (
                    <AlertTriangle className="text-yellow-500 h-5 w-5" />
                  )}
                  <h3 className="font-bold text-lg tracking-tight">
                    {results.viability.status}
                  </h3>
                </div>
                <p className="text-sm opacity-90">
                  {results.viability.message}
                </p>
              </div>

              {/* 3. ESTADÍSTICAS */}
              <div className="grid grid-cols-3 gap-2">
                <div className="bg-muted/30 p-3 rounded-lg text-center border border-border">
                  <span className="block text-xs text-muted-foreground mb-1">
                    Suelo Firme
                  </span>
                  <span className="text-xl font-mono font-semibold">
                    {results.viability.composition.suelo}%
                  </span>
                </div>
                <div className="bg-muted/30 p-3 rounded-lg text-center border border-border">
                  <span className="block text-xs text-muted-foreground mb-1">
                    Arena
                  </span>
                  <span className="text-xl font-mono font-semibold">
                    {results.viability.composition.arena}%
                  </span>
                </div>
                <div className="bg-muted/30 p-3 rounded-lg text-center border border-border">
                  <span className="block text-xs text-muted-foreground mb-1">
                    Obstáculos
                  </span>
                  <span className="text-xl font-mono font-semibold">
                    {results.viability.composition.rocas}%
                  </span>
                </div>
              </div>
            </div>
          )}
        </CardBody>
      </Card>
    </div>
  );
}
