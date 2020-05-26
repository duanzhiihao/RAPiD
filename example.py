from api import Detector

# Initialize detector
detector = Detector(model_name='rapid',
                    weights_path='./weights/pL1_H1MW1024_Mar11_4000.ckpt')

# A simple example to run on a single image and plt.imshow() it
detector.detect_one(img_path='./images/exhibition.jpg',
                    input_size=1024, conf_thres=0.3,
                    visualize=True)
