
import streamlit as st
import torch
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as T
import os

# --- Re-include all necessary definitions for PennFudanDataset, transforms, collate_fn ---
# These are necessary for the model definition if it relies on dataset properties
# or for future expansion, though not directly used by Streamlit's file uploader.
class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, ds, transforms=None):
        self.ds = ds
        self.transforms = transforms

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img_pil = self.ds[idx]["image"]
        mask_pil = self.ds[idx]["label"]

        mask = np.array(mask_pil)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx], dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img_pil, target)
        else:
            img = T.ToImage()(img_pil)
            img = T.ToDtype(torch.float32, scale=True)(img)

        return img, target

def get_transform(train):
    transforms_list = []
    transforms_list.append(T.ToImage())
    transforms_list.append(T.ToDtype(torch.float32, scale=True))

    if train:
        transforms_list.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms_list)

def collate_fn(batch):
    return tuple(zip(*batch))

# --- Define run_inference_on_image function ---
def run_inference_on_image(pil_image, model, device, score_threshold=0.7):
    transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True)
    ])

    img_tensor = transform(pil_image)

    img_for_inference = [img_tensor.to(device)]

    model.eval() # Ensure model is in evaluation mode
    with torch.no_grad():
        prediction = model(img_for_inference)[0]

    keep = prediction["scores"] > score_threshold

    boxes = prediction["boxes"][keep].cpu().numpy()
    masks = prediction["masks"][keep].cpu().numpy()
    scores = prediction["scores"][keep].cpu().numpy()
    labels = prediction["labels"][keep].cpu().numpy()

    masks = (masks > 0.5).astype(np.uint8)

    return boxes, masks, scores, labels

# --- Re-define the Mask R-CNN model (as defined and trained in the notebook) ---
# This will load a fresh, untrained model (with pretrained backbone) if not loaded from state
# For a real application, you would load pre-trained weights from a saved file.
@st.cache_resource
def load_mask_rcnn_model():
    num_classes = 2  # 1 (pedestrian) + 1 (background)
    _model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    in_features = _model.roi_heads.box_predictor.cls_score.in_features
    _model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    in_features_mask = _model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    _model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask,
    dim_reduced=hidden_layer,
    num_classes=num_classes)

    # Move model to device
    _device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    _model.to(_device)
    _model.eval() # Set to evaluation mode by default
    return _model, _device

model, device = load_mask_rcnn_model()

# --- Streamlit Application Layout ---
st.title("Mask R-CNN Pedestrian Instance Segmentation")
st.write("Upload an image to perform instance segmentation using a pre-trained Mask R-CNN model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Original Image", use_column_width=True)
        st.write("")
        st.write("Performing inference...")

        # Run inference
        boxes, masks, scores, labels = run_inference_on_image(image, model, device, score_threshold=0.5)

        # Display results
        if len(boxes) == 0:
            st.write("No pedestrians detected with confidence > 0.5.")
        else:
            fig, ax = plt.subplots(1, figsize=(10, 10))
            ax.imshow(image)
            ax.set_title("Predicted Masks and Bounding Boxes")
            ax.axis("off")

            colors = plt.colormaps.get_cmap("tab10")
            instance_data = []

            for j in range(len(boxes)):
                box = boxes[j]
                mask = masks[j, 0] # Mask is (1, H, W) -> (H, W)
                score = scores[j]
                label = labels[j]

                color = colors(j % 10)

                # Create a semi-transparent colored mask overlay
                colored_mask = np.zeros((*mask.shape, 4), dtype=np.float32)
                colored_mask[mask == 1] = [*color[:3], 0.5] # RGB + Alpha
                ax.imshow(colored_mask)

                # Draw bounding box
                xmin, ymin, xmax, ymax = box
                rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     fill=False, edgecolor=color, linewidth=2)
                ax.add_patch(rect)

                # Store instance data
                instance_data.append({
                    "Instance ID": j + 1,
                    "Label": "pedestrian", # Assuming only pedestrian class for simplicity
                    "Confidence": f"{{score:.2f}}",
                    "Box (xmin, ymin, xmax, ymax)": f"({{xmin:.0f}}, {{ymin:.0f}}, {{xmax:.0f}}, {{ymax:.0f}})"
                })

            st.pyplot(fig)

            if instance_data:
                st.subheader("Detected Instance Statistics")
                st.dataframe(instance_data)

    except Exception as e:
        st.error(f"Error processing image: {{e}}")

