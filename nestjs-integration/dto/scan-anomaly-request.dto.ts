import { IsString, IsBoolean, IsNumber, IsNotEmpty, ValidateNested, IsISO8601 } from 'class-validator';
import { Type } from 'class-transformer';

class ScanMetadataDto {
  @IsNumber()
  latitude: number;

  @IsNumber()
  longitude: number;

  @IsString()
  @IsNotEmpty()
  deviceId: string;
}

class TicketContextDto {
  @IsString()
  @IsNotEmpty()
  ticketId: string;

  @IsString()
  @IsNotEmpty()
  passengerId: string;

  @IsNumber()
  fareAmount: number;

  @IsISO8601()
  purchasedAt: string;

  @IsISO8601()
  expiresAt: string;

  @IsBoolean()
  qrSignatureValid: boolean;
}

class StopContextDto {
  @IsString()
  @IsNotEmpty()
  id: string;

  @IsNumber()
  latitude: number;

  @IsNumber()
  longitude: number;
}

export class ScanAnomalyRequestDto {
  @IsString()
  @IsNotEmpty()
  eventId: string;

  @IsString()
  @IsNotEmpty()
  result: string; // VALID | EXPIRED | ALREADY_USED | INVALID_SIGNATURE

  @IsBoolean()
  isOffline: boolean;

  @IsISO8601()
  scannedAt: string;

  @IsISO8601()
  syncedAt: string;

  @IsNumber()
  syncDelaySeconds: number;

  @ValidateNested()
  @Type(() => ScanMetadataDto)
  scanMetadata: ScanMetadataDto;

  @ValidateNested()
  @Type(() => TicketContextDto)
  ticketContext: TicketContextDto;

  @ValidateNested()
  @Type(() => StopContextDto)
  boardingStop: StopContextDto;
}

export class ScanAnomalyResponseDto {
  eventId: string;
  anomalyScore: number;
  severity: string; // LOW | MEDIUM | HIGH
  reasons: string[];
}
