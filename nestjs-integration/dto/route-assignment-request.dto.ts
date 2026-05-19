import { IsString, IsArray, IsISO8601, ArrayMinSize } from 'class-validator';

export class RouteAssignmentRequestDto {
  @IsString()
  routeId: string;

  @IsISO8601()
  scheduledFor: string;

  @IsArray()
  @IsString({ each: true })
  @ArrayMinSize(1)
  candidateDriverIds: string[];
}

export class DriverSuggestionDto {
  driverId: string;
  driverName: string;
  confidence: number;
  reasons: string[];
}

export class RouteAssignmentResponseDto {
  routeId: string;
  suggestions: DriverSuggestionDto[];
}
