#!/usr/bin/env python

# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""
This module contains the temporal encoders used in the TaMmFSiTS achitecture
"""

import torch
from einops import parse_shape, rearrange, repeat

from tamrfsits.components.missing_data_imputer import MissingDataImputer
from tamrfsits.components.positional_encoder import (
    FixedPositionalEncoder,
    LearnablePositionalEncoder,
)
from tamrfsits.core.time_series import MonoModalSITS
from tamrfsits.core.utils import strip_masks


def append_sensor_token(sits: MonoModalSITS, token: torch.Tensor):
    """
    Append sensor token to featrure dimension
    """
    return MonoModalSITS(
        torch.cat(
            [
                sits.data,
                repeat(
                    token,
                    "c -> b t c w h",
                    b=sits.data.shape[0],
                    t=sits.data.shape[1],
                    w=sits.data.shape[3],
                    h=sits.data.shape[4],
                ),
            ],
            dim=2,
        ),
        doy=sits.doy,
        mask=sits.mask,
    )


class SITSTransformerEncoder(torch.nn.Module):
    """
    A temporal encoder based on the pytorch transformer layer implementation
    """

    def __init__(
        self,
        feature_dim: int = 64,
        nb_heads: int = 2,
        dim_feedforward: int = 256,
        nb_layers: int = 1,
        dropout: float = 0.1,
    ):
        """
        Constructor
        """
        super().__init__()
        self.transformer_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=nb_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=nb_layers,
        )
        self.transformer_encoder.compile(dynamic=True)

    def forward(self, input_sits: MonoModalSITS) -> MonoModalSITS:
        """
        Forward method
        """
        # Compute keys
        input_shape = parse_shape(input_sits.data, "b t c w h")
        input_data = rearrange(input_sits.data, "b t c w h -> (b w h) t c")

        # Apply keys mask if available
        input_mask: torch.Tensor | None = None
        if input_sits.mask is not None:
            input_mask = rearrange(input_sits.mask, "b t w h -> (b w h) t")
        data = self.transformer_encoder(src=input_data, src_key_padding_mask=input_mask)

        data = rearrange(data, "(b w h) t c -> b t c w h", **input_shape)

        return MonoModalSITS(data, input_sits.doy)


class SITSMultiHeadAttention(torch.nn.Module):
    """
    A temporal encoder based on attention
    """

    def __init__(
        self,
        feature_dim: int,
        number_of_heads: int = 1,
        dropout: float = 0.1,
        dim_feedforward: int = 256,
    ):
        """
        Constructor
        """
        super().__init__()

        self.norm1 = torch.nn.LayerNorm(feature_dim)
        self.norm2 = torch.nn.LayerNorm(feature_dim)
        self.linear1 = torch.nn.Linear(feature_dim, dim_feedforward)
        self.linear2 = torch.nn.Linear(dim_feedforward, feature_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.attention = torch.nn.MultiheadAttention(
            feature_dim,
            number_of_heads,
            kdim=feature_dim,
            vdim=feature_dim,
            batch_first=True,
            add_zero_attn=True,
            dropout=dropout,
        )

    def forward(
        self, keys_values_sits: MonoModalSITS, queries_sits: MonoModalSITS | None = None
    ) -> MonoModalSITS:
        """
        Forward method
        """
        input_shape = parse_shape(keys_values_sits.data, "b t c w h")
        keys_values = rearrange(keys_values_sits.data, "b t c w h -> (b w h) t c")

        if queries_sits is not None:
            queries = rearrange(queries_sits.data, "b t c w h -> (b w h) t c")
        else:
            queries_sits = keys_values_sits
            queries = keys_values

        # Build attention mask if required
        key_padding_mask: torch.Tensor | None = None

        if keys_values_sits.mask is not None:
            key_padding_mask = rearrange(
                keys_values_sits.mask,
                "b t w h -> (b w h) t",
            )

        # attn_mask = repeat(key_padding_mask, "b t1 -> b t1 t2", t2=queries.size(-2))
        # scale_factor = 1 / math.sqrt(queries.size(-1))
        # attn_weight = queries @ keys.transpose(-2, -1) * scale_factor
        # attn_weight[attn_mask] = float("-inf")
        # attn_weight = torch.softmax(attn_weight, dim=-1)
        # output_data = attn_weight @ values

        output_data, _ = self.attention(
            queries,
            keys_values,
            keys_values,
            need_weights=False,
            key_padding_mask=key_padding_mask,
        )
        output_data = self.dropout(output_data)

        output_data = self.norm1(queries + output_data)

        output_data = self.norm2(
            output_data
            + self.dropout(
                self.linear2(
                    self.dropout(torch.nn.functional.relu(self.linear1(output_data)))
                )
            )
        )

        if key_padding_mask is not None:
            valid_interp = (~key_padding_mask).sum(dim=1) > 0
            output_data[~valid_interp, :] = 0.0

        keys_output_mask: torch.Tensor | None = None
        output_mask: torch.Tensor | None = queries_sits.mask
        if keys_values_sits.mask is not None:
            keys_output_mask = repeat(
                (~keys_values_sits.mask).sum(dim=1) == 0,
                "b w h -> b t w h",
                t=output_data.shape[1],
            )
            if output_mask is not None:
                output_mask = torch.logical_or(output_mask, keys_output_mask)
            else:
                output_mask = keys_output_mask

        output_data = rearrange(
            output_data,
            "(b w h) t c -> b t c w h",
            w=input_shape["w"],
            h=input_shape["h"],
        )

        return MonoModalSITS(output_data, doy=queries_sits.doy, mask=output_mask)


class TemporalEncoder(torch.nn.Module):
    """
    Class for transformer interpolation
    """

    def __init__(
        self,
        hr_encoder: torch.nn.Module,
        lr_encoder: torch.nn.Module,
        temporal_positional_encoder: (
            FixedPositionalEncoder | LearnablePositionalEncoder
        ),
        token_size: int,
        sensor_token_size: int,
        nb_heads: int = 2,
        dim_feedforward: int = 256,
        nb_layers: int = 1,
        dropout: float = 0.1,
    ):
        """
        Initializer
        """
        super().__init__()

        self.hr_encoder = hr_encoder
        self.lr_encoder = lr_encoder

        self.token_size = token_size
        self.sensor_token_size = sensor_token_size

        # Positional encoder
        self.temporal_positional_encoder = temporal_positional_encoder

        # Learnable tokens for sensor
        self.hr_sensor_token = torch.nn.parameter.Parameter(
            torch.rand(sensor_token_size)
        )
        self.lr_sensor_token = torch.nn.parameter.Parameter(
            torch.rand(sensor_token_size)
        )

        complete_token_size = (
            self.temporal_positional_encoder.get_embedding_size(token_size)
            + sensor_token_size
        )
        self.transformer_encoder = SITSTransformerEncoder(
            feature_dim=complete_token_size,
            nb_heads=nb_heads,
            dim_feedforward=dim_feedforward,
            nb_layers=nb_layers,
            dropout=dropout,
        )

    def forward(
        self,
        lr_sits: MonoModalSITS | None,
        hr_sits: MonoModalSITS | None,
    ) -> MonoModalSITS:
        """
        Implementation of the forward method
        """
        # We assume that masked have been stripped already
        assert lr_sits is None or lr_sits.mask is None
        assert hr_sits is None or hr_sits.mask is None

        # Then, encode sits
        hr_encoded = self.hr_encoder(hr_sits) if hr_sits else None
        lr_encoded = self.lr_encoder(lr_sits) if lr_sits else None

        # Apply positional encoding
        hr_encoded = (
            self.temporal_positional_encoder(hr_encoded) if hr_encoded else None
        )
        lr_encoded = (
            self.temporal_positional_encoder(lr_encoded) if lr_encoded else None
        )

        # Add sensor token
        hr_encoded = (
            append_sensor_token(hr_encoded, self.hr_sensor_token)
            if hr_encoded
            else None
        )
        lr_encoded = (
            append_sensor_token(lr_encoded, self.lr_sensor_token)
            if lr_encoded
            else None
        )

        # Finally, merge into a single sits
        tokens = MonoModalSITS(
            torch.cat(
                [s.data for s in (lr_encoded, hr_encoded) if s is not None], dim=1
            ),
            torch.cat(
                [s.doy for s in (lr_encoded, hr_encoded) if s is not None], dim=1
            ),
        )

        transformed = self.transformer_encoder(tokens)

        return transformed


class TemporalDecoder(torch.nn.Module):
    """
    Class for transformer interpolation
    """

    def __init__(
        self,
        decoder: torch.nn.Module,
        temporal_positional_encoder: FixedPositionalEncoder,
        nb_heads: int,
        token_size: int,
        sensor_token_size: int,
        dropout: float = 0.1,
    ):
        """
        Initializer
        """
        super().__init__()

        self.decoder = decoder
        self.token_size = token_size
        self.sensor_token_size = sensor_token_size

        # Learnable token for missing data
        self.imputer = MissingDataImputer(nb_features=token_size, learnable_token=True)

        # Positional encoder
        self.temporal_positional_encoder = temporal_positional_encoder

        # Learnable tokens for sensor
        self.query_sensor_token = torch.nn.parameter.Parameter(
            torch.rand(sensor_token_size)
        )

        complete_token_size = (
            self.temporal_positional_encoder.get_embedding_size(token_size)
            + sensor_token_size
        )

        self.attention = SITSMultiHeadAttention(
            complete_token_size, nb_heads, dropout=dropout
        )

    def generate_query_sits(
        self, query_doy: torch.Tensor, source_sits: MonoModalSITS
    ) -> MonoModalSITS:
        """
        Generate query sits with positional encoding and sensor token
        """
        if len(query_doy.shape) == 1:
            query_doy = repeat(query_doy, "t -> b t", b=source_sits.data.shape[0])
        assert len(query_doy.shape) == 2
        data = torch.zeros(
            (
                source_sits.data.shape[0],
                query_doy.shape[1],
                self.token_size,
                source_sits.data.shape[3],
                source_sits.data.shape[4],
            ),
            dtype=source_sits.data.dtype,
            device=source_sits.data.device,
        )
        mask = torch.full(
            (
                source_sits.data.shape[0],
                query_doy.shape[1],
                source_sits.data.shape[3],
                source_sits.data.shape[4],
            ),
            True,
            dtype=torch.bool,
            device=source_sits.data.device,
        )

        return MonoModalSITS(data, query_doy, mask)

    def forward(
        self,
        encoded_sits: MonoModalSITS,
        query_doy: torch.Tensor,
    ) -> MonoModalSITS:
        """
        Implementation of the forward method
        """
        # Then, encode sits

        # Process queries
        # LR encoded SITS is only passed as a convenience to get correct tensor shapes
        query = self.generate_query_sits(query_doy, encoded_sits)
        query = strip_masks(self.imputer(query))
        query = self.temporal_positional_encoder(query)
        query = append_sensor_token(query, self.query_sensor_token)

        transformed = self.attention(encoded_sits, query)

        # Apply datewise convolutional decoder
        output = self.decoder(transformed)

        return output
