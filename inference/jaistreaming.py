import cv2
import torch
import eBUS as eb
import argparse
from utils import util
import numpy as np
import lib.PvSampleUtils as psu
from PvStreamSample import configure_stream_buffers, configure_stream
from time import time

opencv_is_available = True
# model = torch.load('./weights/X_1280d.pt','cuda')['model'].float().fuse()
model = torch.load('./weights/allData/X/X_1280d(dataset0+parts_Val,v0-4+Aug_Train).pt','cuda')['model'].float().fuse()
# model = torch.load('./weights/v8_s.pt', 'cpu')['model'].float().fuse()

model.half()
model.eval()


class Camera:
    def __init__(self, args, ids):

        self.connection_ID = []
        self.BUFFER_COUNT = 1
        self.kb = psu.PvKb()

        self.device = []
        self.buffer_list = []
        self.stream = []
        self.doodle = "|\\-|-/"
        self.doodle_index = [0, 0, 0, 0, 0]
        self.frame_rate = []
        self.bandwidth = []
        self.stop = []

        for idx in range(len(ids)):
            # print(len(ids))
            self.connection_ID.append(id)
            self.device.append(self.connect_to_device(ids[idx]))

            if self.device[idx]:
                self.stream.append(self.open_stream(ids[idx]))
                if self.stream[idx]:
                    # configure the stream based on Gig or USB connection
                    self.configure_stream(idx)
                    # set the
                    self.buffer_list.append(self.configure_stream_buffers(idx))
                    # acquire the images
                    self.setup_cams_buffer(idx)
                    # self.acquire_images()

            else:
                print("Error: Cant open device !")

        self.acquire_images(args)

    def connect_to_device(self, connection_ID):
        # Connect to the GigE Vision or USB3 Vision device
        print("Connecting to device.")
        result, device = eb.PvDevice.CreateAndConnect(connection_ID)
        if device is None:
            print(
                f"Unable to connect to device: {result.GetCodeString()} ({result.GetDescription()})")
        return device

    def open_stream(self, connection_ID):
        # Open stream to the GigE Vision or USB3 Vision device
        print("Opening stream from device.")
        result, stream = eb.PvStream.CreateAndOpen(connection_ID)
        if stream is None:
            print(
                f"Unable to stream from device. {result.GetCodeString()} ({result.GetDescription()})")
        return stream

    def configure_stream(self, idx):
        # If this is a GigE Vision device, configure GigE Vision specific streaming parameters
        if isinstance(self.device[idx], eb.PvDeviceGEV):
            # Negotiate packet size
            self.device[idx].NegotiatePacketSize()
            # Configure device streaming destination
            self.device[idx].SetStreamDestination(
                self.stream[idx].GetLocalIPAddress(), self.stream[idx].GetLocalPort())

    def configure_stream_buffers(self, idx):
        buffer_list = []
        # Reading payload size from device
        size = self.device[idx].GetPayloadSize()

        # Use BUFFER_COUNT or the maximum number of buffers, whichever is smaller
        buffer_count = self.stream[idx].GetQueuedBufferMaximum()
        buffer_count = min(buffer_count, self.BUFFER_COUNT)
        # if buffer_count > self.BUFFER_COUNT:
        #     buffer_count = self.BUFFER_COUNT

        # Allocate buffers
        for _ in range(buffer_count):
            # Create new pvbuffer object
            pvbuffer = eb.PvBuffer()
            # Have the new pvbuffer object allocate payload memory
            pvbuffer.Alloc(size)
            # Add to external list - used to eventually release the buffers
            buffer_list.append(pvbuffer)

        # Queue all buffers in the stream
        for pvbuffer in buffer_list:
            self.stream[idx].QueueBuffer(pvbuffer)
        # print(f"Created {buffer_count} buffers")

        return buffer_list

    def setup_cams_buffer(self, idx):
        # Get device parameters need to control streaming
        device_params = self.device[idx].GetParameters()

        # Map the GenICam AcquisitionStart and AcquisitionStop commands
        start = device_params.Get("AcquisitionStart")
        self.stop.append(device_params.Get("AcquisitionStop"))

        # Get stream parameters
        stream_params = self.stream[idx].GetParameters()

        # Map a few GenICam stream stats counters
        self.frame_rate.append(stream_params.Get("AcquisitionRate"))
        self.bandwidth.append(stream_params["Bandwidth"])

        # Enable streaming and send the AcquisitionStart command
        print("Enabling streaming and sending AcquisitionStart command.")
        self.device[idx].StreamEnable()
        start.Execute()

    def acquire_images(self, args):
        display_image = False
        warning_issued = False

        # Acquire images until the user instructs us to stop.
        # print("\n<press a key to stop streaming>")
        self.kb.start()
        while not self.kb.is_stopping():
            # Retrieve next pvbuffer
            for idx in range(len(self.connection_ID)):
                # print("IDX : ", idx)

                t1 = time()
                result, pvbuffer, operational_result = self.stream[idx].RetrieveBuffer(1)
                # print(f"CAM : {idx} Buffer retreival time : {time() - t1}")
                if result.IsOK():
                    if operational_result.IsOK():

                        result, frame_rate_val = self.frame_rate[idx].GetValue()
                        result, bandwidth_val = self.bandwidth[idx].GetValue()

                        if 0:
                            print(f"{self.doodle[self.doodle_index[idx]]} BlockID: {pvbuffer.GetBlockID()}", end='')

                        payload_type = pvbuffer.GetPayloadType()
                        if 0:
                            print(f"payload_type: {payload_type}")
                            print(f"eb.PvPayloadTypeImage: {eb.PvPayloadTypeImage}")
                        if payload_type == eb.PvPayloadTypeImage:
                            image = pvbuffer.GetImage()
                            image_data = image.GetDataPointer()
                            if 0:
                                print(f" W: {image.GetWidth()} H: {image.GetHeight()} ", end='')

                            if opencv_is_available:
                                # print(f"the image type of {idx} is: {image.GetPixelType()}")
                                if image.GetPixelType() == eb.PvPixelMono8:
                                    display_image = True
                                if image.GetPixelType() == eb.PvPixelRGB8:
                                    image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
                                    display_image = True

                                if display_image:
                                    # print("GRABED THE FRAME from !!", idx)
                                    # cv2.namedWindow(f'stream-{idx}', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                                    # cv2.imshow(f'stream-{idx}',cv2.resize(image_data, (640,480)))
                                    x = np.stack([util.resize(image_data, args.input_size)], axis=0)
                                    # x = x[:, :, np.newaxis]
                                    x = x.transpose((0, 3, 1, 2))  # HWC to CHW
                                    x = np.ascontiguousarray(x)  # contiguous
                                    x = torch.from_numpy(x).cuda()
                                    x = x.half()  # uint8 to fp16/32
                                    x = x / 255.  # 0 - 255 to 0.0 - 1.0
                                    # Inference
                                    output = model(x)
                                    output = util.non_max_suppression(output, 0.1, 0.5, model.head.nc)
                                    shape = image_data.shape
                                    centers = []
                                    # Scale outputs
                                    gain = min(x.shape[2] / shape[0], x.shape[3] /shape[1])  # gain  = old / new
                                    pad = (x.shape[3] - shape[1] * gain) /2, (x.shape[2] - shape[0]*gain) / 2  # wh padding

                                    # x padding
                                    output = output[0]
                                    output[:, [0, 2]] -= pad[0]
                                    output[:, [1, 3]] -= pad[1]
                                    output[:, :4] /= gain

                                    output[:, 0].clamp_(0, shape[1])  # x1
                                    output[:, 1].clamp_(0, shape[0])  # y1
                                    output[:, 2].clamp_(0, shape[1])  # x2
                                    output[:, 3].clamp_(0, shape[0])  # y2
                                    # Draw boxes
                                    num_holes = 0
                                    image_data = np.stack((image_data, image_data, image_data), axis=-1)
                                    for box in output:
                                        x1, y1, x2, y2 = list(map(int, box[:4]))
                                        center = [(x1+x2)/2, (y1+y2)/2]
                                        if int(box.cpu().numpy()[5]) == 0:
                                            num_holes += 1
                                            cv2.rectangle(image_data, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                        else:
                                            cv2.rectangle(image_data, (x1, y1), (x2, y2), (255, 0, 255), 2)
                                        centers.append(center)
                                    cv2.putText(image_data,f'Number of Holes: {num_holes}',(50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                                    h, w = image_data.shape[:2]
                                    h, w = 3 * h // 4, 3 * w // 4
                                    image_data = cv2.resize(image_data, (w, h))
                                    # cv2.imwrite(os.path.join(OUT_DIR, filename), image)
                                    util.display("result", image_data)
                                    if cv2.waitKey(27) & 0xFF == ord("q"):
                                        break
                                else:
                                    if not warning_issued:
                                        # display a message that video only display for Mono8 / RGB8 images
                                        print(f" ")
                                        print(f" Currently only Mono8 / RGB8 images are displayed", end='\r')
                                        print(f"")
                                        warning_issued = True

                                if cv2.waitKey(1) & 0xFF != 0xFF:
                                    cv2.destroyAllWindows()
                                    break

                        elif payload_type == eb.PvPayloadTypeChunkData:
                            print(f" Chunk Data payload type with {pvbuffer.GetChunkCount()} chunks", end='')

                        elif payload_type == eb.PvPayloadTypeRawData:
                            print(f" Raw Data with {pvbuffer.GetRawData().GetPayloadLength()} bytes", end='')

                        elif payload_type == eb.PvPayloadTypeMultiPart:
                            print(f" Multi Part with {pvbuffer.GetMultiPartContainer().GetPartCount()} parts", end='')

                        else:
                            print(" Payload type not supported by this sample", end='')

                        print(f" {frame_rate_val:.1f} FPS  {bandwidth_val / 1000000.0:.1f} Mb/s     ", end='\r')
                    else:
                        # Non OK operational result
                        print(f"{self.doodle[ self.doodle_index[idx] ]} {operational_result.GetCodeString()}       ", end='\r')
                    # Re-queue the pvbuffer in the stream object
                    self.stream[idx].QueueBuffer(pvbuffer)

                else:
                    # Retrieve pvbuffer failure
                    print(f"{self.doodle[ self.doodle_index[idx] ]} {result.GetCodeString()}      ", end='\r')

                self.doodle_index[idx] = (self.doodle_index[idx] + 1) % 6
                if self.kb.kbhit():
                    self.kb.getch()
                    break

                # if len(self.frame_buffer) > 10:
                #     sleep(0.05)

        self.kb.stop()
        if opencv_is_available:
            cv2.destroyAllWindows()

        # Tell the device to stop sending images.
        for idx in range(len(self.connection_ID)):
            print("\nSending AcquisitionStop command to the device")
            self.stop[idx].Execute()

            # Disable streaming on the device
            print("Disable streaming on the controller.")
            self.device[idx].StreamDisable()

            # Abort all buffers from the stream and dequeue
            print("Aborting buffers still in stream")
            self.stream[idx].AbortQueuedBuffers()
            while self.stream[idx].GetQueuedBufferCount() > 0:
                result, pvbuffer, lOperationalResult = self.stream.RetrieveBuffer()

        return

    def disconnect_streams(self):
        self.buffer_list.clear()

        # Close the stream
        print("Closing stream")
        self.stream.Close()
        eb.PvStream.Free(self.stream)

        # Disconnect the device
        print("Disconnecting device")
        self.device.Disconnect()
        eb.PvDevice.Free(self.device)

    def display_images(self):
        while True:
            print("Thread 2")
            if len(self.frame_buffer) > 0:
                self.thread_lock.acquire()
                frame = self.frame_buffer.pop(0)
                self.thread_lock.release()

                cv2.imshow(f'Frame-{self.id}', frame)
                cv2.waitKey(1)

# # connection_ID = ['169.254.251.42', '169.254.251.43']#, '169.254.251.43']
# t0 = time()
# print(f"initial time : {time()-t0}")
# connection_ID = ['192.168.10.101', '192.168.10.102','192.168.10.103', '192.168.10.104']#, '10.169.31.147', '10.169.31.148'] #, '10.169.31.149']

# cam1 = Camera(connection_ID)

# cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=1280, type=int  )
    args = parser.parse_args()

    util.setup_seed()
    util.setup_multi_processes()
    connection_ID = ['169.254.17.193']
    Camera(args, connection_ID)


if __name__ == "__main__":
    main()
