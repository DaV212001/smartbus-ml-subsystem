# SmartBus ML Subsystem — NestJS Core Integration Guide

This guide outlines how to seamlessly integrate the **SmartBus ML microservice (FastAPI)** with the core **NestJS backend application**. 

By utilizing relative paths and standard environment variables, this integration setup is completely system-agnostic and ready for GitHub deployment.

---

## 🏛️ Schema Alignment

Your NestJS Prisma schema and local PostgreSQL enums match the ML simulated data directly:

* **TripStatus:** `SCHEDULED` | `IN_PROGRESS` | `COMPLETED` | `CANCELLED`
* **TicketStatus:** `ACTIVE` | `USED`
* **ScanResult:** `VALID` | `EXPIRED` | `ALREADY_USED` | `INVALID_SIGNATURE` | `INSPECTION_ONLY`

---

## 🔧 Step-by-Step Integration

### Step 1: Add the NestJS Module
1. Copy the generated files from `smartbus-ml-subsystem/nestjs-integration/` into your NestJS project:
   * Place the files under your NestJS source tree: `src/modules/ml/`
2. Create `src/modules/ml/ml.module.ts` to register the controller and service:

```typescript
import { Module } from '@nestjs/common';
import { HttpModule } from '@nestjs/axios';
import { SmartBusMlController } from './smartbus-ml.controller';
import { SmartBusMlService } from './smartbus-ml.service';

@Module({
  imports: [HttpModule],
  controllers: [SmartBusMlController],
  providers: [SmartBusMlService],
  exports: [SmartBusMlService],
})
export class MlModule {}
```

3. **Install the HTTP client wrapper dependencies** inside your NestJS root directory:
   ```bash
   npm install @nestjs/axios axios
   ```

4. **Register the module** in your main `src/app.module.ts`:
   Add `MlModule` to the `imports: [...]` array of your core module.

5. **Set the environment variable** in your NestJS `.env` file:
   ```env
   ML_SERVICE_URL=http://localhost:8000
   ```

---

### Step 2: Integrate AI Route Driver Recommendations (Feature 1)

In your **`trips.controller.ts`** (or relevant administrative scheduling controller), inject the `SmartBusMlService` to suggest the best drivers for scheduled routes:

```typescript
import { Controller, Post, Body } from '@nestjs/common';
import { SmartBusMlService } from '../ml/smartbus-ml.service';

@Controller('trips')
export class TripsController {
  constructor(
    private readonly tripsService: TripsService,
    private readonly mlService: SmartBusMlService,
  ) {}

  @Post('assign-suggestions')
  async getDriverSuggestions(
    @Body() dto: { routeId: string; scheduledFor: string; candidateDriverIds: string[] }
  ) {
    return this.mlService.getRouteAssignmentSuggestions(dto);
  }
}
```

---

### Step 3: Integrate Ticket Scan Anomaly Auditing (Feature 2)

In your **`validation.service.ts`**, you can automatically dispatch scan audits to the ML model in the background or during sync operations:

```typescript
import { Injectable, Logger } from '@nestjs/common';
import { SmartBusMlService } from '../ml/smartbus-ml.service';

@Injectable()
export class ValidationService {
  private readonly logger = new Logger(ValidationService.name);

  constructor(
    private prisma: PrismaService,
    private tripsService: TripsService,
    private mlService: SmartBusMlService,
  ) {}

  async validateTicket(driverId: string, dto: ValidateTicketDto) {
    // 1. Perform signature checks, ticket lookup, and DB updates...
    const finalResult = ScanResult.VALID;
    
    // 2. Dispatch the ML anomaly check asynchronously (Non-blocking)
    try {
      const boardingStop = await this.prisma.stop.findFirst({
        where: { route: { trips: { some: { id: tripId } } } }
      });

      this.mlService.detectScanAnomaly({
        eventId: `EV-${Date.now()}-${ticket.id}`,
        result: finalResult,
        isOffline: false,
        scannedAt: scannedAt.toISOString(),
        syncedAt: new Date().toISOString(),
        syncDelaySeconds: 0,
        scanMetadata: {
          latitude: dto.latitude ?? 9.025, // Extracted from client boarding GPS
          longitude: dto.longitude ?? 38.765,
          deviceId: dto.deviceId ?? 'DEV-5021',
        },
        ticketContext: {
          ticketId: ticket.id,
          passengerId: ticket.passengerId,
          fareAmount: Number(ticket.fareAmount),
          purchasedAt: ticket.purchasedAt.toISOString(),
          expiresAt: ticket.expiresAt.toISOString(),
          qrSignatureValid: true,
        },
        boardingStop: {
          id: boardingStop?.id ?? 'BS-UNKNOWN',
          latitude: Number(boardingStop?.latitude ?? 9.024),
          longitude: Number(boardingStop?.longitude ?? 38.764),
        }
      }).then(auditResult => {
        if (auditResult.severity === 'HIGH') {
          this.logger.warn(
            `[FRAUD-ALERT] Scan flagged as HIGH anomaly! Reasons: ${auditResult.reasons.join(', ')}`
          );
          // E.g., Notify inspectors, emit socket events, or log to a security/audit table
        }
      });
    } catch (mlErr) {
      this.logger.error('Failed to dispatch background scan anomaly audit:', mlErr);
    }

    return { result: finalResult, ticket: finalTicket, passenger };
  }
}
```

---

## 🛡️ Robust Graceful Degradation Heuristic
If the Python ML microservice becomes unreachable or timed out (1.5-second hard limit), the `SmartBusMlService` handles the failure gracefully. It invokes a **local database aggregation/TS rules fallback** (pre-coded in `smartbus-ml.service.ts`), providing full operational resilience for active bus boarding gates.

---

## 🚀 Render Deployment Guide

To deploy this Python FastAPI microservice to **Render** alongside your NestJS application, follow these steps:

### Step 1: Create a New Web Service on Render
1. Connect your GitHub repository to your Render dashboard.
2. Select the repository containing this microservice (`smartbus-ml-subsystem`).
3. Set the following basic parameters:
   * **Runtime:** `Python 3` (or choose Docker if utilizing the provided `Dockerfile`).
   * **Build Command:**
     ```bash
     pip install -r requirements.txt && python -m src.train
     ```
     > [!IMPORTANT]
     > Running `python -m src.train` during the build step ensures that your synthetic datasets are generated and the trained ML models are baked directly into the immutable deployment slug. This avoids model loading errors on startup and preserves models across Render container restarts.
   * **Start Command:**
     ```bash
     uvicorn src.main:app --host 0.0.0.0 --port $PORT
     ```

### Step 2: Configure Environment Variables on Render
Add the following key-value pairs in the **Environment** tab:
* `PORT`: `8000` (Render will override this dynamically for routing, but it ensures standard fallback bindings).
* `ENV`: `production`
* `TRAINING_SAMPLE_SIZE`: `2000`
* `FRAUD_SAMPLE_SIZE`: `5000`

### Step 3: Link NestJS Backend to Render Service
Once Render assigns your FastAPI web service a public URL (e.g., `https://smartbus-ml.onrender.com`), configure your NestJS environment to route calls to it:

1. Update the `.env` configuration file in your **NestJS repository**:
   ```env
   ML_SERVICE_URL=https://smartbus-ml.onrender.com
   ```
2. Redeploy or restart your NestJS service. The NestJS HTTP adapter will now communicate directly with your live Render-hosted ML engine.

