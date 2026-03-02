### Object detection implementation into Hololens 2 for human-robot cooperation

Using OpenXR 1.14.3 version and MRTK3 Toolkit to make the base of the project. Model will be implemented with Sentis 2 from unity.<br>
**From now on, the work will continue in Setis, as a resreach paper tested between 3 evironments and Sentis (formerly known as Barracuda) came out as the most optimal solution**<br>

Plan and progress of the models:
| Model name | WinML | Unity Sentis | Notes |
|---|---|---|---|
| SSD-MobileNetV1 | 100% | 70% | Works on WinML with stable outcome |
| SSD | 0% | 0% |   |
| Tiny YOLOv3 | 0% | 20% | Needs different opset, Delayed, due to hard to find solutions |
| YOLOv4 | ~~45%~~ | 0% | Solution on WinML is unstable, See if finding a workaround or new solution |
| Faster-RCNN | 0% | 0% |  |
| Mask-RCNN | 0% | 0% |   |
