# Seeding Performance Investigation

## Host snapshot
- Ubuntu 25.04 (`6.14.0-33-generic`) on AMD Ryzen 9 5900X with all cores set to the `powersave` governor (`/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`), which keeps clocks low unless threads demand bursts.
- Load average hovers near `1.3` with long-running `ffmpeg` (PID 2045740) occupying a full core and `qbittorrent-nox` (PID 100741) consuming ~18 % CPU (`top -b -n1`).
- Primary NIC `enp4s0f0` is an Intel ixgbe 10 GbE adapter syncing at `10000Mb/s` full duplex with negligible interface drops (`ethtool enp4s0f0`, `ip -s link show enp4s0f0`).

## Network stack observations
- TCP autotuning headroom is small: `net.core.rmem_max`/`wmem_max` remain at 212,992 B with `tcp_rmem`/`tcp_wmem` max values of 6 MiB/4 MiB, limiting socket buffer growth for high-latency peers (`sysctl` output). This caps one-connection throughput on long fat pipes.
- Congestion control stays at default `cubic`; `tcp_mtu_probing` is disabled (`0`), so MTU issues would force fragmentation instead of probing.
- `netstat -s` reports 14,651 TCP “bad segments received” and thousands of SACK recovery fallbacks, hinting at packet loss or buffer pressure.

## qBittorrent configuration highlights
- Anonymous mode is enabled and all peer-discovery helpers are disabled (`Session\AnonymousModeEnabled=true`, `Session\DHTEnabled=false`, `Session\PeXEnabled=false`, `Session\LSDEnabled=false` in `~/.config/qBittorrent/qBittorrent.conf`). Without DHT/PeX/LSD the client depends solely on trackers, dramatically shrinking the pool of incoming peers for seeding.
- NAT traversal is off: `Network\PortForwardingEnabled=false`, and logs show repeated UPnP/NAT-PMP failures leaving the service without an external port despite the 11739 listener (`/mnt/data/torrent/logs/qbittorrent.log`). Without a forwarded port, the client can accept far fewer inbound connections, throttling seeding.
- `Session\MaxUploadsPerTorrent=400` and `Session\MaxConnections=10000` are generous, but `Session\ConnectionSpeed=100` (half-open throttle) may slow ramp-up when many torrents start simultaneously.
- Tracker filtering is enabled and could discard peers from certain trackers (`Session\TrackerFilteringEnabled=true`).

## Additional considerations
- CPU is not saturated overall (plenty of idle), but the `powersave` governor can delay turbo boosts; switching to `performance` typically helps latency-sensitive workloads.
- Storage resides on large HDD-backed LVM with SSD metadata; there are no obvious capacity constraints (`df -h`, `lsblk`). Disk I/O contention was not observed but worth monitoring during peak seeding.
- Docker bridges expose the BitTorrent port on multiple virtual interfaces; ensure firewall/NAT rules on the upstream router map the chosen external port consistently.

## Potential follow-ups (no changes made)
1. Re-enable peer discovery (DHT, PeX, LSD) and disable anonymous mode if tracker policies allow, to increase the swarm and inbound peers.
2. Restore reliable port forwarding (static mapping of TCP/UDP 11739 or another port) so the client is reachable from the internet.
3. Raise socket buffer ceilings (`net.core.{rmem,wmem}_max`, `net.ipv4.tcp_{rmem,wmem}`) and consider enabling `tcp_mtu_probing`/ECN to better utilize the 1 Gbps link.
4. Switch CPU governor to `performance` or `schedutil` to avoid power-saving latency spikes.
5. Investigate packet loss path (router, ISP, cabling) given the TCP error counters.
