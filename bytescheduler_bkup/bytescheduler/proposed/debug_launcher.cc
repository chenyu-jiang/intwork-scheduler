#include <iostream>
#include <queue>
#include <mutex>
#include <sstream>
#include <condition_variable>

#include "common.h"
#include "controller_mpi.h"

using namespace proposed::common;

// A threadsafe-queue.
template <class T>
class SafeQueue
{
public:
  SafeQueue(): q(), m(), c() {}

  // Add an element to the queue.
  void enqueue(T t)
  {
    std::lock_guard<std::mutex> lock(m);
    q.push(t);
    c.notify_one();
  }

  // Get the "front"-element.
  // If the queue is empty, wait till a element is avaiable.
  T dequeue()
  {
    std::unique_lock<std::mutex> lock(m);
    while(q.empty())
    {
      // release lock as long as the wait and reaquire it afterwards.
      c.wait(lock);
    }
    T val = q.front();
    q.pop();
    return val;
  }

private:
  std::queue<T> q;
  mutable std::mutex m;
  std::condition_variable c;
};

struct SafeQueueElement {
    SafeQueueElement(int32_t i, int32_t p) {
        layer_id = i;
        partition_id = p;
    }
    int32_t layer_id;
    int32_t partition_id;
};

int main() {
    SafeQueue<SafeQueueElement> queue;
    std::vector<std::vector<Tensor>> tensors;

    std::vector<std::vector<Tensor>> small_tensors;

    int32_t layer_num = 5;
    int32_t partition_per_layer = 16;

    int32_t small_tensor_per_layer = 4;

    for(int i=0;i<layer_num;i++) {
      std::vector<Tensor> ts;
      for(int p=0;p<partition_per_layer;p++)
          ts.emplace_back(
              Tensor(i, p, 
              [&queue, i, p](){
                  queue.enqueue(SafeQueueElement(i, p));
                  }));
      tensors.emplace_back(std::move(ts));

      for(int j=0;j<small_tensor_per_layer;j++) {
        std::vector<Tensor> sts;
        sts.emplace_back(Tensor((i+1)*1000+j, 0, 
              [&queue, i, j](){
                  queue.enqueue(SafeQueueElement((i+1)*1000+j, 0));
                  }));
        small_tensors.emplace_back(std::move(sts));
      }
    }

    MPIController controller;
    controller.Initialize();

    for(int layer_id = layer_num-1; layer_id >= 0; layer_id--) {
      controller.PostTensor(tensors[layer_id], layer_id);
      for(int j=0;j<small_tensor_per_layer;j++) {
        // std::cout<<"posting "+std::to_string(small_tensors[layer_id*small_tensor_per_layer+j][0].tensor_id) << std::endl;
        controller.PostTensor(small_tensors[layer_id*small_tensor_per_layer+j], layer_id, 0);
      }
    }

    int32_t my_rank = controller.get_rank();

    controller.LaunchBackGroundThread();
    std::cout << "BG Thread Launched." << std::endl;

    for(int i=0;i<layer_num* (partition_per_layer + small_tensor_per_layer); i++) {
      SafeQueueElement e = queue.dequeue();
      if (e.layer_id >= 1000) {
        std::stringstream stream;
        stream << "[rank " << my_rank << "] Layer " << e.layer_id << " Partition " << e.partition_id << " finished." << std::endl;
        std::cout << stream.str();
      }
      controller.SignalPartitionFinished(e.layer_id, e.partition_id);
    }
    MPI_Finalize();
    return 0;
}