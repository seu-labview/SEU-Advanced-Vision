from yolo6D.Predict import *

if __name__ == "__main__":
    name = 'ZB008'
    img_path = '1.jpg'
    img = cv2.imread(img_path)
    model = dn('yolo6D/yolo-pose.cfg')
    model.load_weights('weights/' + name + '.weights')
    boxes = detect(name, model, img_path)
    best_conf_est = -1
    for j in range(len(boxes)):
        if (boxes[j][18] > best_conf_est):
            box_pr = boxes[j]
            best_conf_est = boxes[j][18]
    
    bss = [np.array(np.reshape(box_pr[:18], [9, 2]), dtype='float32')]
    strs = []
    strs.append(name)
    strs.append(str(int(best_conf_est.numpy() * 100)) + '%')
    strss = [strs]
    draw_predict(bss, strss, img, -1)