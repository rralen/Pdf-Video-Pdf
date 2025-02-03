import cv2
import numpy as np
import struct

def image_to_block(frame, grid_size=64):
    """
    Convierte un fotograma del video en un bloque de 512 bytes.
    Se redimensiona la imagen a grid_size x grid_size usando INTER_NEAREST, 
    se extraen los bits (0 o 1) y se agrupan en bytes.
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    img_small = cv2.resize(gray, (grid_size, grid_size), interpolation=cv2.INTER_NEAREST)
    
    bits = []
    for i in range(grid_size):
        for j in range(grid_size):
            bits.append(1 if img_small[i, j] > 127 else 0)
    
    bytes_list = []
    for i in range(0, len(bits), 8):
        group = bits[i:i+8]
        byte = 0
        for bit in group:
            byte = (byte << 1) | bit
        bytes_list.append(byte)
    return bytes(bytes_list)

def decode_video(video_path, output_file, grid_size=64):
    """
    Lee el video y reconstruye el archivo original.
    Se asume que:
      - El primer fotograma contiene un encabezado:
          * Los primeros 8 bytes representan el tamaño original del PDF.
          * Los siguientes 504 bytes son los primeros bytes del PDF.
      - Cada fotograma codifica un bloque de 512 bytes.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error al abrir el video")
        return

    blocks = []
    frame_count = 0
    filesize = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        block = image_to_block(frame, grid_size=grid_size)
        if frame_count == 0:
            if len(block) < 8:
                print("Error: bloque de encabezado incompleto")
                cap.release()
                return
            filesize = struct.unpack("!Q", block[:8])[0]
            print(f"Tamaño original del archivo (según encabezado): {filesize} bytes")
            blocks.append(block)
        else:
            blocks.append(block)
        frame_count += 1
    cap.release()
    print(f"Procesados {frame_count} fotogramas.")

    # Concatenar todos los bloques
    data = b"".join(blocks)
    if filesize is None:
        print("Error: No se encontró el encabezado en el video.")
        return

    # Se omiten los primeros 8 bytes del encabezado; el PDF original se compone de:
    #   * Los bytes a partir del byte 8 (del primer bloque) hasta alcanzar filesize bytes.
    pdf_data = data[8:8+filesize]

    try:
        with open(output_file, "wb") as f:
            f.write(pdf_data)
        print(f"Archivo decodificado guardado en: {output_file}")
    except Exception as e:
        print(f"Error al escribir el archivo de salida: {e}")

if __name__ == "__main__":
    video_path = "output_video.avi"           # Video generado por el codificador
    output_file = "Archivo_decodificado.pdf"    # PDF reconstruido
    decode_video(video_path, output_file)
