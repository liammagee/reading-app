import { useEffect, useState } from 'react';
import { syncService, type SyncStatus } from './SyncService';

/**
 * React hook that tracks sync status for UI indicators.
 */
export function useSyncStatus(): SyncStatus {
  const [status, setStatus] = useState<SyncStatus>(syncService.status);

  useEffect(() => {
    return syncService.onStatusChange(setStatus);
  }, []);

  return status;
}
