import cv2
import numpy as np
import struct

def pdf_a_binario(pdf_path):
    """Lee el PDF en binario."""
    try:
        with open(pdf_path, "rb") as f:
            return f.read()
    except Exception as e:
        print(f"Error al leer el PDF: {e}")
        return None

def block_to_image(block, grid_size=64):
    """
    Convierte un bloque de 512 bytes en una imagen (cuadrícula) de tamaño grid_size x grid_size.
    Cada byte se transforma en 8 bits y se mapea a 0 (negro) o 255 (blanco).
    """
    bits = []
    for byte in block:
        # Convertir cada byte en 8 bits (de más significativo a menos)
        bits.extend([ (byte >> i) & 1 for i in reversed(range(8)) ])
    total_bits = grid_size * grid_size
    if len(bits) < total_bits:
        bits.extend([0] * (total_bits - len(bits)))
    arr = np.array(bits, dtype=np.uint8) * 255
    return arr.reshape((grid_size, grid_size))

def generar_video(pdf_path, video_path, frame_size=(640, 640), grid_size=64, fps=10.0):
    """
    Genera un video a partir de un PDF utilizando bloques reversibles de 512 bytes.
    El primer bloque contiene un encabezado de 8 bytes (tamaño del PDF) + 504 bytes de datos.
    Se utiliza el códec "MJPG" en contenedor AVI para minimizar la compresión.
    """
    pdf_data = pdf_a_binario(pdf_path)
    if pdf_data is None:
        return

    filesize = len(pdf_data)
    block_size = grid_size * grid_size // 8  # 4096 bits / 8 = 512 bytes

    # Preparar el bloque de encabezado (8 bytes de tamaño + 504 bytes de datos)
    header = struct.pack("!Q", filesize)  # 8 bytes big-endian
    remaining = block_size - len(header)  # 504 bytes
    header_block = header + pdf_data[:remaining]

    # Crear la lista de bloques
    blocks = [header_block]
    offset = remaining
    while offset < filesize:
        block = pdf_data[offset: offset + block_size]
        if len(block) < block_size:
            block += b'\x00' * (block_size - len(block))
        blocks.append(block)
        offset += block_size

    # Configurar VideoWriter con códec MJPG y contenedor AVI
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
    if not out.isOpened():
        print("Error al abrir el VideoWriter")
        return

    for block in blocks:
        img = block_to_image(block, grid_size=grid_size)
        # Escalar la imagen de 64x64 a frame_size (en este ejemplo, 640x640) usando INTER_NEAREST
        img_resized = cv2.resize(img, frame_size, interpolation=cv2.INTER_NEAREST)
        frame = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        out.write(frame)
    out.release()
    print(f"Video generado en: {video_path}")

if __name__ == "__main__":
    pdf_path = "Archivo.pdf"         # PDF de entrada
    video_path = "output_video.avi"    # Video de salida (AVI)
    generar_video(pdf_path, video_path)
