import serial, struct, zmq

SBUS_START = 0x0F
SBUS_FRAME = 25
SBUS_MIN, SBUS_MAX = 272, 1712
SBUS_RANGE = SBUS_MAX - SBUS_MIN
FMT = "<13f"  # 13 little-endian floats

def decode(frame):
    d = struct.unpack("<25B", frame)
    ch = [0]*13
    ch[0]  = ( d[1]     | d[2]<<8) & 0x07FF
    ch[1]  = ((d[2]>>3  | d[3]<<5) & 0x07FF)
    ch[2]  = ((d[3]>>6  | d[4]<<2 | d[5]<<10) & 0x07FF)
    ch[3]  = ((d[5]>>1  | d[6]<<7) & 0x07FF)
    ch[4]  = ((d[6]>>4  | d[7]<<4) & 0x07FF)
    ch[5]  = ((d[7]>>7  | d[8]<<1 | d[9]<<9 ) & 0x07FF)
    ch[6]  = ((d[9]>>2  | d[10]<<6) & 0x07FF)
    ch[7]  = ((d[10]>>5 | d[11]<<3) & 0x07FF)
    ch[8]  = ( d[12]    | d[13]<<8) & 0x07FF
    ch[9]  = ((d[13]>>3 | d[14]<<5) & 0x07FF)
    ch[10] = ((d[14]>>6 | d[15]<<2 | d[16]<<10) & 0x07FF)
    ch[11] = ((d[16]>>1 | d[17]<<7) & 0x07FF)
    ch[12] = ((d[17]>>4 | d[18]<<4) & 0x07FF)
    norm = [max(-1.0, min(1.0, 2.0 * ((v - SBUS_MIN) / SBUS_RANGE) - 1.0)) for v in ch]
    return norm, bool(d[23] & 0x04), bool(d[23] & 0x08)

def read_frames(ser):
    while True:
        b = ser.read(1)
        if not b: 
            continue
        if b[0] == SBUS_START:
            rest = ser.read(SBUS_FRAME - 1)
            if len(rest) == SBUS_FRAME - 1:
                yield b + rest

def main(port="/dev/ttyUSB0", baud=115200, bind="tcp://*:8011", topic=b"sbus"):
    ser = serial.Serial(port, baud, timeout=0.02)
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.bind(bind)
    try:
        for frame in read_frames(ser):
            norm, frame_lost, failsafe = decode(frame)
            payload = struct.pack(FMT, *norm)
            header = b"\x01" if (frame_lost or failsafe) else b"\x00"
            pub.send_multipart([topic, header, payload])
    finally:
        pub.close(0)
        ser.close()
        ctx.term()

if __name__ == "__main__":
    # change args here if needed
    main(port="/dev/ttyUSB0", baud=115200, bind="tcp://*:8011", topic=b"sbus")
