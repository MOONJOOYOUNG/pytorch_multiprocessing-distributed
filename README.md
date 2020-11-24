# pytorch_multiprocessing-distributed

args.world_size = 사용할 gpu 수

## 동작
* 사용할 gpu 수만큼 초기 init weight가 같은 네트워크 구성.
  * 배치가 128면, 각 네트워크당 배치크기 : 128 / gpu 수
* data sampler를 통해 서로 다른 네트워크에 서로 다른 데이터 배분.
  * 배치가 128고 사용하는 gpu가 4개라면 iter 마다 128개의 데이터를 보게됨. 
* 각 네트워크마다 그레이디언트를 구하고 각각 값을 모아서 평균을 구함.
* 평균을 각 네트워크에 뿌려주고 네트워크의 모델을 업데이트함
* 결과 정리 시 get_rank()를 이용해서 하나의 gpu 값만 확인하고 record 하면됌   
``` 
# 0 : 0번 gpu를 말함
if dist.get_rank() == 0:
        logger.write([epoch, losses.avg, top1.avg, top5.avg])
``` 
