import zmq, struct
FMT = "<13f"
ctx = zmq.Context.instance()
sub = ctx.socket(zmq.SUB)
sub.connect("tcp://localhost:8011")
sub.setsockopt(zmq.SUBSCRIBE, b"sbus")
while True:
    topic, header, payload = sub.recv_multipart()
    flags = payload and header != b"\x00"
    vals = struct.unpack(FMT, payload)
    print(vals)
