# Global settings of the environment
import os

from dotenv import load_dotenv

# ==============================
# CSV FILES OF COLLECTED FINGERPRINTS
# ==============================

# loads environment
current_folder = os.path.dirname(os.path.abspath(__file__))
CONFIG_FOLDER = os.path.join(current_folder, "../../config")
load_dotenv(os.path.join(CONFIG_FOLDER, "folder_paths.config"))

EVALUATION_CSV_FOLDER_PATH =  os.getenv("csv_folder") + "/evaluation"
TRAINING_CSV_FOLDER_PATH = os.getenv("csv_folder") + "/training"
FINGERPRINT_FOLDER_PATH = os.getenv("saved_fingerprints_folder")
ALL_CSV_HEADERS = "time,timestamp,seconds,connectivity,cpu_us,cpu_sy,cpu_ni,cpu_id,cpu_wa,cpu_hi,cpu_si,tasks_total,tasks_running,tasks_sleeping,tasks_stopped,tasks_zombie,mem_free,mem_used,mem_cache,swap_avail,net_eth_rx,net_eth_tx,tcp_counter,tcp_out,rx_drop,tcp_retrans,cpu_temp,alarmtimer:alarmtimer_fired,alarmtimer:alarmtimer_start,block:block_bio_backmerge,block:block_bio_remap,block:block_dirty_buffer,block:block_getrq,block:block_touch_buffer,block:block_unplug,clk:clk_set_rate,cpu-migrations,cs,dma_fence:dma_fence_init,fib:fib_table_lookup,filemap:mm_filemap_add_to_page_cache,gpio:gpio_value,ipi:ipi_raise,irq:irq_handler_entry,irq:softirq_entry,jbd2:jbd2_handle_start,jbd2:jbd2_start_commit,kmem:kfree,kmem:kmalloc,kmem:kmem_cache_alloc,kmem:kmem_cache_free,kmem:mm_page_alloc,kmem:mm_page_alloc_zone_locked,kmem:mm_page_free,kmem:mm_page_pcpu_drain,mmc:mmc_request_start,net:net_dev_queue,net:net_dev_xmit,net:netif_rx,page-faults,pagemap:mm_lru_insertion,qdisc:qdisc_dequeue,qdisc:qdisc_dequeue,raw_syscalls:sys_enter,raw_syscalls:sys_exit,rpm:rpm_resume,rpm:rpm_suspend,sched:sched_process_exec,sched:sched_process_free,sched:sched_process_wait,sched:sched_switch,sched:sched_wakeup,signal:signal_deliver,signal:signal_generate,skb:consume_skb,skb:consume_skb,skb:kfree_skb,skb:kfree_skb,skb:skb_copy_datagram_iovec,sock:inet_sock_set_state,task:task_newtask,tcp:tcp_destroy_sock,tcp:tcp_probe,timer:hrtimer_start,timer:timer_start,udp:udp_fail_queue_rcv_skb,workqueue:workqueue_activate_work"
DUPLICATE_HEADERS = ["qdisc:qdisc_dequeue", "skb:consume_skb", "skb:kfree_skb"]

# ==============================
# RASPBERRY CLIENT
# ==============================

IP_DEVICE_5555 = "192.168.191.242"
IP_DEVICE_5556 = "192.168.191.212"
# add as many devices as needed ...

CLIENT_DEVICES = [IP_DEVICE_5555, IP_DEVICE_5556]

LIVE_TRAINING_DEVICE = IP_DEVICE_5555

# ==============================
# ANOMALY DETECTION
# ==============================

MAX_ALLOWED_CORRELATION_IF = 0.99
MAX_ALLOWED_CORRELATION_AE = 0.98
DROP_CONNECTIVITY = ["connectivity"]
DROP_TEMPORAL = ["time", "timestamp", "seconds"]
DROP_CONSTANT = ['cpu_ni', 'cpu_hi', 'tasks_stopped', "rx_drop", "tcp_retrans", 'alarmtimer:alarmtimer_fired', 'alarmtimer:alarmtimer_start',
                 'dma_fence:dma_fence_init', "qdisc:qdisc_dequeue", 'udp:udp_fail_queue_rcv_skb']
DROP_UNSTABLE = ['cpu_wa', 'mem_free', 'mem_used', 'mem_cache', 'swap_avail', 'cpu_temp', 'filemap:mm_filemap_add_to_page_cache', 'ipi:ipi_raise',
                 'jbd2:jbd2_start_commit', 'kmem:kfree', 'kmem:kmalloc', 'kmem:mm_page_alloc_zone_locked',
                 'kmem:mm_page_pcpu_drain', 'net:net_dev_queue', 'net:net_dev_xmit', 'qdisc:qdisc_dequeue',
                 'rpm:rpm_suspend', 'skb:consume_skb', 'tcp:tcp_probe', 'timer:timer_start',
                 'workqueue:workqueue_activate_work']

# ==============================
# C2 SIMULATION
# ==============================

MAX_STEPS_V2 = 1000

MAX_EPISODES_V3 = 250
MAX_STEPS_V3 = 20  # avg 4.5 steps

MAX_EPISODES_V4 = 10_000
SIM_CORPUS_SIZE_V4 = 4000  # 4000 for 20 1s steps with 200 bytes/s

MAX_EPISODES_V5 = 10_000
SIM_CORPUS_SIZE_V5 = 4000  # 4000 for 8 steps with 500 bytes/s

MAX_EPISODES_V6 = 10_000
SIM_CORPUS_SIZE_V6 = 4000  # 4000 for 8 steps with 500 bytes/s

MAX_EPISODES_V7 = 10_000
SIM_CORPUS_SIZE_V7 = 4000  # 4000 for 8 steps with 500 bytes/s

MAX_EPISODES_V8 = 300
SIM_CORPUS_SIZE_V8 = 4000  # 4000 for 8 steps with 500 bytes/s

MAX_EPISODES_V9 = 1000
SIM_CORPUS_SIZE_V9 = 4000  # 4000 for 8 steps with 500 bytes/s

MAX_EPISODES_V10 = 100
SIM_CORPUS_SIZE_V10 = 4000  # 4000 for 8 steps with 500 bytes/s

MAX_EPISODES_V98 = 5000

MAX_STEPS_V99 = 500
