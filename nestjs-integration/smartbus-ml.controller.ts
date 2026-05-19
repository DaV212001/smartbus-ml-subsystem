import { Controller, Post, Body, HttpCode, HttpStatus, UsePipes, ValidationPipe } from '@nestjs/common';
import { SmartBusMlService } from './smartbus-ml.service';
import { 
  RouteAssignmentRequestDto, 
  RouteAssignmentResponseDto 
} from './dto/route-assignment-request.dto';
import { 
  ScanAnomalyRequestDto, 
  ScanAnomalyResponseDto 
} from './dto/scan-anomaly-request.dto';

@Controller('ml')
export class SmartBusMlController {
  constructor(private readonly mlService: SmartBusMlService) {}

  /**
   * HTTP POST Endpoint to get driver route assignment suggestions.
   * Dispatches to the internal Python FastAPI ML microservice.
   * Path: POST /ml/route-assignment
   */
  @Post('route-assignment')
  @HttpCode(HttpStatus.OK)
  @UsePipes(new ValidationPipe({ transform: true, whitelist: true }))
  async getRouteAssignmentSuggestions(
    @Body() payload: RouteAssignmentRequestDto
  ): Promise<RouteAssignmentResponseDto> {
    return this.mlService.getRouteAssignmentSuggestions(payload);
  }

  /**
   * HTTP POST Endpoint to evaluate a passenger QR scan transaction.
   * Performs an immediate hybrid rule/Isolation Forest outlier check.
   * Path: POST /ml/detect-anomaly
   */
  @Post('detect-anomaly')
  @HttpCode(HttpStatus.OK)
  @UsePipes(new ValidationPipe({ transform: true, whitelist: true }))
  async detectScanAnomaly(
    @Body() payload: ScanAnomalyRequestDto
  ): Promise<ScanAnomalyResponseDto> {
    return this.mlService.detectScanAnomaly(payload);
  }
}
